"""Map IP addresses to organizations using WHOIS records (e.g., from
whois.arin.net).

This uses a global database of addresses and does not update existing records
(we assume network records are static enough relative to when we query).
This is mostly to minimize rate limiting."""

import ipaddress
import logging
import re
import time
from typing import Optional

import igraph as ig
import ipwhois
import sqlite3
import tqdm

import config
import parse.graph
import util

log = logging.getLogger(__name__)

def addr_from_sockname(name: str) -> Optional[str]:
    """Return just the IP address from a graph node name for a socket."""
    addr = name[5:]
    # assume IPv4 addresses
    try:
        addr, _ = addr.rsplit(":", maxsplit=1)
        return addr
    except:
        #log.warning("warning: unrecognized address %s", addr)
        return None

def default_record(ip: str):
    """Return a default record if no (useful) data can be found."""
    log.warning("%s: assuming no record", ip)
    name = "\\0no record" # NUL doesn't work well with sqlite3
    cidr = f"{ip}/32"
    cidr = ipaddress.ip_network(cidr)
    return name, cidr

def get_name(con: sqlite3.Connection, db: sqlite3.Cursor, ip: str) -> str:
    """Lookup `ip` in `db` and return the corresponding network name.
    This will either use a cached record, or perform a WHOIS query and cache
    the result."""
    # skip empty or unknown
    if not ip or ip.startswith(parse.graph.unknown_prefix):
        return ip

    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        log.error("%s: not an address?", ip)
        return ip

    if addr.version != 4:
        # hardcode a few common address ranges
        reserved = {
            "::/128": "unspecified address",
            "::1/128": "loopback",
            "::ffff:0:0/96": "IPv4 mapped",
            "::ffff:0:0:0/96": "IPv4 translated",
            "64:ff9b::/96": "IPv4 translation",
            "64:ff9b:1::/48": "IPv4 translation",
            "100::/64": "discard",
            "2001:0000::/32": "Teredo tunneling",
            "2001:20::/28": "ORCHIDv2",
            "2001:db8::/32": "Example",
            "2002::/16": "6to4",
            "fc00::/7": "unique local",
            "fe80::/10": "link-local",
            "ff00::/8": "multicast",
        }
        for k, v in reserved.items():
            if addr in ipaddress.ip_network(k):
                return v

        # otherwise, ignore IPv6 for now
        log.warning("ignoring IPv6 address: %s", ip)
        return ip

    res = db.execute("SELECT name "
                     "FROM whois "
                     "WHERE (? & mask) = subnet "
                     "ORDER BY mask DESC", (int(addr),))
    res = res.fetchone()
    if res is not None:
        return res[0]

    try:
        # library is sometimes fragile...
        success = False
        timeout = 2
        while not success:
            try:
                whois = ipwhois.IPWhois(ip).lookup_rdap()
                success = True

                # lots of fields here, but ASN doesn't seem to always be filled out
                # whereas network is much more reliable
                name = whois["network"]["name"]
                cidr = whois["network"]["cidr"]

                # sometimes get "NA"; check ASN cidr first
                if cidr == "NA":
                    try:
                        cidr = ipaddress.ip_network(whois["asn_cidr"])
                        name = whois["asn_description"]
                    except ValueError:
                        name, cidr = default_record(ip)
                # sometimes, the network reports multiple CIDRs. pick the better one
                elif "," in cidr:
                    try:
                        best_cidr = ipaddress.ip_network(whois["asn_cidr"])
                        name = whois["asn_description"]
                    except ValueError:
                        best_cidr = ipaddress.ip_network("0.0.0.0/0")
                    for c in cidr.split(", "):
                        c = ipaddress.ip_network(c)
                        if addr not in c:
                            continue
                        if c.prefixlen >= best_cidr.prefixlen:
                            best_cidr = c
                            name = whois["network"]["name"]
                    cidr = best_cidr
                    # make sure we actually found something...
                    if addr not in cidr:
                        name, cidr = default_record(ip)
                else:
                    try:
                        cidr = ipaddress.ip_network(cidr)
                        if addr not in cidr:
                            raise ValueError
                    except ValueError:
                        name, cidr = default_record(ip)
            except ipwhois.exceptions.HTTPLookupError:
                # maybe try waiting?
                log.warning("lookup error; waiting for %s seconds...", timeout)
                time.sleep(timeout)
                timeout *= 2
                if timeout > 30:
                    # seems like there actually is no record
                    # (e.g., 198.98.23.55)
                    name, cidr = default_record(ip)
                    break
    except ipwhois.exceptions.IPDefinedError:
        # https://en.wikipedia.org/wiki/Reserved_IP_addresses
        reserved = {
            "0.0.0.0/8": "current network",
            "10.0.0.0/8": "private network",
            "100.64.0.0/10": "shared address space",
            "127.0.0.0/8": "loopback",
            "169.254.0.0/16": "link-local",
            "172.16.0.0/12": "private network",
            "192.0.0.0/24": "IETF Protocol Assignments",
            "192.0.2.0/24": "TEST-NET-1",
            "192.88.99.0/24": "6to4 relay",
            "192.168.0.0/16": "private network",
            "198.18.0.0/15": "inter-network benchmarking",
            "198.51.100.0/24": "TEST-NET-2",
            "203.0.113.0/24": "TEST-NET-3",
            "224.0.0.0/4": "multicast",
            "240.0.0.0/4": "reserved",
            "255.255.255.255": "limited broadcast",
        }
        cidr = None
        for k, v in reserved.items():
            if addr in ipaddress.ip_network(k):
                cidr = k
                name = v
        if cidr is None:
            raise
        cidr = ipaddress.ip_network(cidr)

    subnet = int(cidr.network_address)
    mask = int(cidr.netmask)
    if name is None:
        log.warning("%s: no name found! using %s as name...", ip, cidr)
        name = str(cidr)

    db.execute("INSERT INTO whois (subnet, mask, name) "
               "VALUES (?, ?, ?)", (subnet, mask, name))
    con.commit()
    return name

def convert_sockets(g: ig.Graph) -> None:
    log.info("renaming IP addresses based on WHOIS records...")
    con = sqlite3.connect(f"{config.graph_transforms.ip_whois}/cache.sqlite3")
    db = con.cursor()
    # since ipwhois reports network CIDRs, we can further cache results by
    # looking up whether an IP falls within a CIDR or not
    db.execute("CREATE TABLE IF NOT EXISTS whois ("
               "  subnet INTEGER NOT NULL,"
               "  mask INTEGER NOT NULL,"
               "  name TEXT NOT NULL,"
               "  PRIMARY KEY (subnet, mask)"
               ")")
    con.commit()

    for u in tqdm.tqdm(g.vs(type_eq="socket")):
        addr = addr_from_sockname(u["name"])
        if addr is None:
            continue
        u["name"] = get_name(con, db, addr)

    con.commit()
    con.close()
