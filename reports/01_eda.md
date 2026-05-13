# Sumo Data EDA Report

## Bashos
- rows: **60**
- range: `201501` … `202411`

## Bouts
- rows: **17,586**
- basho range: `201501` … `202411`
- unique basho: 59
- days covered: 1 … 15

- valid bouts (winner present, east+west ids non-zero): **17,586**
- east-win rate (trivial baseline): **0.5024**
- kimarite filled rate: **1.0000**

### Top-10 kimarite

| kimarite | count | share |
|---|---:|---:|
| `yorikiri` | 4,606 | 0.262 |
| `oshidashi` | 4,133 | 0.235 |
| `hatakikomi` | 1,561 | 0.089 |
| `tsukiotoshi` | 1,203 | 0.068 |
| `uwatenage` | 709 | 0.040 |
| `hikiotoshi` | 635 | 0.036 |
| `tsukidashi` | 633 | 0.036 |
| `oshitaoshi` | 526 | 0.030 |
| `okuridashi` | 507 | 0.029 |
| `yoritaoshi` | 449 | 0.026 |

### Bouts per basho
- min: 266  max: 316  median: 300

## Rikishis
- rows: **679**
- height non-null: 679 (1.000)
- weight non-null: 679 (1.000)
- birthDate non-null: 679
- unique heya: 51

### Anthropometry
- height (cm): mean=179.3 std=6.8
- weight (kg): mean=140.2 std=25.2

## Banzuke
- rows (rikishi×basho): **2,477**
- rankValue non-null: 2,477 (1.000)
- unique rikishi: 121

### Most common rank labels

| rank | count |
|---|---:|
| `Yokozuna 1 East` | 59 |
| `Maegashira 7 West` | 59 |
| `Sekiwake 1 West` | 59 |
| `Maegashira 1 West` | 59 |
| `Maegashira 2 West` | 59 |
| `Maegashira 3 West` | 59 |
| `Maegashira 4 West` | 59 |
| `Maegashira 5 West` | 59 |
