"""Thin abstraction over Binance Futures REST/WebSocket APIs."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from binance import AsyncClient  # type: ignore

from .config import Settings

logger = logging.getLogger(__name__)


@dataclass
class ExchangeBalance:
    asset: str
    free: float
    cross_wallet_balance: float
    available_balance: float


class ExchangeClient:
    """Manage authenticated connectivity with Binance Futures."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._client: Optional[AsyncClient] = None

    async def _ensure_client(self) -> AsyncClient:
        if self._client is None:
            if not self.settings.binance_api_key or not self.settings.binance_api_secret:
                raise RuntimeError("Binance credentials are required for live trading")
            self._client = await AsyncClient.create(
                api_key=self.settings.binance_api_key,
                api_secret=self.settings.binance_api_secret,
                testnet=self.settings.mode != "live",
            )
        return self._client

    async def get_exchange_time(self) -> datetime:
        client = await self._ensure_client()
        payload = await client.futures_time()
        return datetime.fromtimestamp(payload["serverTime"] / 1000)

    async def fetch_balance(self) -> ExchangeBalance:
        client = await self._ensure_client()
        info = await client.futures_account_balance()
        usdt = next((row for row in info if row["asset"] == "USDT"), None)
        if usdt is None:
            raise RuntimeError("USDT balance missing")
        return ExchangeBalance(
            asset="USDT",
            free=float(usdt.get("withdrawAvailable", 0.0)),
            cross_wallet_balance=float(usdt.get("crossWalletBalance", 0.0)),
            available_balance=float(usdt.get("availableBalance", 0.0)),
        )

    async def set_leverage(self, symbol: str, leverage: int) -> None:
        client = await self._ensure_client()
        await client.futures_change_leverage(symbol=symbol, leverage=leverage)

    async def place_order(self, params: Dict[str, Any]) -> Dict[str, Any]:
        client = await self._ensure_client()
        logger.info("placing order", extra={"params": params})
        response = await client.futures_create_order(**params)
        return response

    async def cancel_order(self, symbol: str, order_id: int | None, client_order_id: str | None) -> Dict[str, Any]:
        client = await self._ensure_client()
        return await client.futures_cancel_order(symbol=symbol, orderId=order_id, origClientOrderId=client_order_id)

    async def fetch_open_orders(self, symbol: str | None = None) -> List[Dict[str, Any]]:
        client = await self._ensure_client()
        return await client.futures_get_open_orders(symbol=symbol)

    async def fetch_positions(self) -> List[Dict[str, Any]]:
        client = await self._ensure_client()
        return await client.futures_position_information()

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close_connection()
            self._client = None


async def bootstrap_margin(settings: Settings, exchange: ExchangeClient) -> None:
    """Ensure isolated margin and leverage are applied for each symbol."""

    client = await exchange._ensure_client()
    for symbol in settings.universe:
        try:
            await client.futures_change_margin_type(symbol=symbol, marginType="ISOLATED")
            await exchange.set_leverage(symbol, int(settings.leverage))
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("failed to configure margin", extra={"symbol": symbol, "error": str(exc)})
