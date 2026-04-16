# Bot po prodazhe santekhniki

Proekt pod chat-bota, kotoryi:
- khranit katalog santekhniki i reklamnye tovary;
- raspoznayet intenty polzovatelya cherez ML;
- podbiraet tovary iz kataloga;
- otvechaet fallback-replikami;
- mozhet byt podklyuchen k Telegram.

## Struktura

- `data/catalog/` - katalog tovarov i reklamnykh pozitsii
- `data/intents/` - intenty, primery fraz i otvety
- `data/dialogues/` - rezervnye/fallback otvety
- `data/ads/` - reklamnyye scenarii po tovaram
- `src/bot/core/` - obshchaya logika bota
- `src/bot/ml/` - obuchenie i inference ML-modeli
- `src/bot/services/` - rabota s katalogom i podborom
- `src/bot/telegram/` - integratsiya s Telegram
- `src/bot/utils/` - ochistka teksta i normalizatsiya
- `models/` - sokhranennye modeli
- `tests/` - testy dialogov, podbora i reklamy
- `docs/` - dokumentatsiya i roadmap

