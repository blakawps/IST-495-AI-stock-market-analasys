import re

from db import companies


EXCHANGE_ALIASES = {
    "nasdaq": "NASDAQ",
    "nasdaq global market": "NASDAQ",
    "nasdaq capital market": "NASDAQ",
    "nasdaq global select market": "NASDAQ",

    "nyse": "NYSE",
    "nyse american": "NYSE AMERICAN",
    "nyse arca": "NYSE ARCA",
    "amex": "NYSE AMERICAN",

    "otc markets": "OTC",
    "other otc": "OTC",
    "otcqx": "OTCQX",
    "otcqb": "OTCQB",

    "tsx": "TSX",
    "tsx-v": "TSX-V",
    "tsx venture": "TSX-V",

    "lse": "LSE",
    "london": "LSE",

    "amsterdam": "EURONEXT AMSTERDAM",
}


LEGAL_SUFFIXES = {
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "company",
    "co",
    "limited",
    "ltd",
    "plc",
    "llc",
}

ALLOWED_EXCHANGES = {
    "NASDAQ",
    "NYSE",
    "NYSE AMERICAN",
}

def normalize_exchange(exchange):
    if not exchange:
        return None

    exchange = exchange.strip()

    return EXCHANGE_ALIASES.get(
        exchange.lower(),
        exchange.upper()
    )


def normalize_text(text):
    """
    Normalize text so company names can be compared
    safely regardless of punctuation/capitalization.
    """

    if not text:
        return ""

    text = text.casefold()

    # AT&T -> at and t
    text = text.replace("&", " and ")

    # Replace punctuation with spaces
    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def remove_legal_suffix(company_name):
    """
    Apple Inc -> Apple
    NVIDIA Corporation -> NVIDIA
    Ford Motor Company -> Ford Motor
    """

    normalized = normalize_text(
        company_name
    )

    if not normalized:
        return ""

    parts = normalized.split()

    while (
        len(parts) > 1
        and parts[-1] in LEGAL_SUFFIXES
    ):
        parts.pop()

    return " ".join(parts)


def valid_ticker(symbol):
    if not symbol:
        return False

    return bool(
        re.fullmatch(
            r"[A-Z][A-Z0-9.\-]{0,9}",
            symbol.upper()
        )
    )


class TickerResolver:

    def __init__(self):

        self.company_names = {}
        self.token_index = {}
        self.known_symbols = set()
        self.symbol_exchange = {}

        self.load_companies()

    # -------------------------------------------------
    # COMPANY DATABASE
    # -------------------------------------------------

    def load_companies(self):

        cursor = companies.find(
            {
                "enabled": True,
                "country_code": "US",
                "asset_type": "Stock",
            },
            {
                "_id": 0,
                "symbol": 1,
                "exchange": 1,
                "company_name": 1,
                "aliases": 1,
                "manual_aliases": 1,
            }
        )

        company_count = 0
        name_count = 0

        for company in cursor:

            symbol = (
                company.get("symbol", "")
                .strip()
                .upper()
            )

            exchange = normalize_exchange(
                company.get("exchange")
            )

            if exchange not in ALLOWED_EXCHANGES:
                continue

            company_name = (
                company.get(
                    "company_name",
                    ""
                ).strip()
            )

            if not symbol or not company_name:
                continue

            self.known_symbols.add(symbol)
            self.symbol_exchange[symbol] = exchange

            company_count += 1

            # -----------------------------------------
            # Official full company name
            # -----------------------------------------

            self.register_name(
                company_name,
                symbol,
                exchange,
                confidence=96,
                name_type="official_name"
            )

            # -----------------------------------------
            # Remove legal suffix
            #
            # Apple Inc -> Apple
            # NVIDIA Corporation -> NVIDIA
            # -----------------------------------------

            simplified_name = (
                remove_legal_suffix(
                    company_name
                )
            )

            if (
                simplified_name
                and normalize_text(
                    simplified_name
                )
                != normalize_text(
                    company_name
                )
            ):

                self.register_name(
                    simplified_name,
                    symbol,
                    exchange,
                    confidence=92,
                    name_type="simplified_name"
                )

            # -----------------------------------------
            # CSV aliases
            # -----------------------------------------

            for alias in company.get(
                "aliases",
                []
            ):

                self.register_name(
                    alias,
                    symbol,
                    exchange,
                    confidence=90,
                    name_type="alias"
                )

            # -----------------------------------------
            # Manually-added aliases
            # -----------------------------------------

            for alias in company.get(
                "manual_aliases",
                []
            ):

                self.register_name(
                    alias,
                    symbol,
                    exchange,
                    confidence=94,
                    name_type="manual_alias"
                )

        for names in self.token_index.values():
            name_count += len(names)

        print(
            f"Ticker resolver loaded "
            f"{company_count} companies."
        )

    def register_name(
        self,
        name,
        symbol,
        exchange,
        confidence,
        name_type
    ):

        normalized_name = normalize_text(
            name
        )

        if not normalized_name:
            return

        # Ignore extremely short company aliases.
        # Examples such as "A" or "IT" are too dangerous.
        if len(normalized_name) < 3:
            return

        security = {
            "symbol": symbol,
            "exchange": exchange,
            "source": "company_name",
            "confidence": confidence,
            "name_type": name_type,
            "matched_name": name,
        }

        if normalized_name not in self.company_names:

            self.company_names[
                normalized_name
            ] = []

        self.company_names[
            normalized_name
        ].append(
            security
        )

        # -----------------------------------------
        # Build token index for faster searching
        #
        # "nvidia corporation"
        # first token = "nvidia"
        # -----------------------------------------

        first_token = (
            normalized_name
            .split()[0]
        )

        if first_token not in self.token_index:

            self.token_index[
                first_token
            ] = set()

        self.token_index[
            first_token
        ].add(
            normalized_name
        )

        # -----------------------------------------
        # Companies beginning with "The"
        #
        # "The Trade Desk"
        # also allow
        # "Trade Desk"
        # -----------------------------------------

        if normalized_name.startswith(
            "the "
        ):

            without_the = (
                normalized_name[4:]
            )

            if len(without_the) >= 3:

                if without_the not in self.company_names:

                    self.company_names[
                        without_the
                    ] = []

                self.company_names[
                    without_the
                ].append(
                    security
                )

                first_token = (
                    without_the
                    .split()[0]
                )

                if first_token not in self.token_index:

                    self.token_index[
                        first_token
                    ] = set()

                self.token_index[
                    first_token
                ].add(
                    without_the
                )

    # -------------------------------------------------
    # DIRECT RSS / HEADLINE TICKER EXTRACTION
    # -------------------------------------------------

    def extract_direct_tickers(
        self,
        entry,
        title
    ):

        securities = []
        seen = set()

        def add_security(
            symbol,
            exchange=None,
            source=None,
            confidence=100
        ):

            if not symbol:
                return

            symbol = (
                symbol.strip()
                .upper()
                .replace("$", "")
            )

            if not valid_ticker(symbol):
                return

            normalized_exchange = (
                normalize_exchange(
                    exchange
                )
                if exchange
                else None
            )

            if normalized_exchange is None:
                normalized_exchange = (
                    self.symbol_exchange.get(symbol)
                )

            # Only allow NASDAQ, NYSE, and AMEX.
            if normalized_exchange not in ALLOWED_EXCHANGES:
                return

            key = (
                normalized_exchange or "",
                symbol
            )

            if key in seen:
                return

            seen.add(key)

            securities.append({
                "symbol": symbol,
                "exchange": normalized_exchange,
                "source": source,
                "confidence": confidence,
            })

        # -----------------------------------------
        # 1. RSS metadata
        #
        # GlobeNewswire example:
        #
        # Nasdaq:NVDA
        # NYSE:F
        # -----------------------------------------

        for tag in entry.get(
            "tags",
            []
        ):

            term = str(
                tag.get(
                    "term",
                    ""
                )
            ).strip()

            if ":" not in term:
                continue

            exchange_text, symbol = (
                term.split(
                    ":",
                    1
                )
            )

            exchange_key = (
                exchange_text
                .strip()
                .lower()
            )

            if (
                exchange_key
                not in EXCHANGE_ALIASES
            ):
                continue

            add_security(
                symbol=symbol,
                exchange=exchange_text,
                source="rss_metadata",
                confidence=100
            )

        # -----------------------------------------
        # 2. NASDAQ:AAPL / NYSE:F
        # -----------------------------------------

        exchange_pattern = re.compile(
            r"\b("
            r"NYSE\s+American|"
            r"NYSE\s+Arca|"
            r"OTC\s+Markets|"
            r"NASDAQ|"
            r"NYSE|"
            r"AMEX|"
            r"OTCQX|"
            r"OTCQB|"
            r"TSX-V|"
            r"TSX|"
            r"LSE"
            r")"
            r"\s*:\s*"
            r"\$?"
            r"([A-Z][A-Z0-9.\-]{0,9})\b",
            re.IGNORECASE
        )

        for match in (
            exchange_pattern
            .finditer(title)
        ):

            add_security(
                symbol=match.group(2),
                exchange=match.group(1),
                source="headline_exchange_ticker",
                confidence=100
            )

        # -----------------------------------------
        # 3. $AAPL
        # -----------------------------------------

        dollar_pattern = re.compile(
            r"\$([A-Z][A-Z0-9.\-]{0,9})\b"
        )

        for match in (
            dollar_pattern
            .finditer(title)
        ):

            symbol = (
                match.group(1)
                .upper()
            )

            # Only trust $SYMBOL if it exists
            # in our company database.
            if symbol not in self.known_symbols:
                continue

            add_security(
                symbol=symbol,
                exchange=None,
                source="headline_dollar_ticker",
                confidence=95
            )

        return securities

    # -------------------------------------------------
    # COMPANY NAME RESOLUTION
    # -------------------------------------------------

    def resolve_company_names(
        self,
        title
    ):

        normalized_title = (
            normalize_text(title)
        )

        if not normalized_title:
            return []

        title_tokens = set(
            normalized_title.split()
        )

        candidate_names = set()

        # -----------------------------------------
        # Only search company names whose first
        # word actually exists in the headline.
        #
        # Much faster than checking every company.
        # -----------------------------------------

        for token in title_tokens:

            candidates = (
                self.token_index.get(
                    token
                )
            )

            if candidates:

                candidate_names.update(
                    candidates
                )

        padded_title = (
            f" {normalized_title} "
        )

        matches = {}

        # Longer company names first.
        #
        # Example:
        # "Meta Platforms" before "Meta"
        #
        for company_name in sorted(
            candidate_names,
            key=len,
            reverse=True
        ):

            phrase = (
                f" {company_name} "
            )

            if phrase not in padded_title:
                continue

            for security in (
                self.company_names.get(
                    company_name,
                    []
                )
            ):

                key = (
                    security.get(
                        "exchange"
                    ) or "",
                    security["symbol"]
                )

                existing = (
                    matches.get(key)
                )

                # Keep highest-confidence match
                # if several aliases matched.
                if (
                    existing is None
                    or security[
                        "confidence"
                    ]
                    > existing[
                        "confidence"
                    ]
                ):

                    resolved = (
                        security.copy()
                    )

                    resolved[
                        "matched_name"
                    ] = company_name

                    matches[key] = (
                        resolved
                    )

        return list(
            matches.values()
        )

    # -------------------------------------------------
    # MAIN RESOLVER
    # -------------------------------------------------

    def resolve(
        self,
        entry,
        title
    ):

        # First trust actual ticker information.
        direct_tickers = (
            self.extract_direct_tickers(
                entry,
                title
            )
        )

        if direct_tickers:
            return direct_tickers

        # If ticker wasn't explicitly provided,
        # try company-name resolution.
        return self.resolve_company_names(
            title
        )