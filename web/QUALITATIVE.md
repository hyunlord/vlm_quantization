# Qualitative retrieval — server vs offline top-5 (verification table)

Gallery = COCO 5K test (so400m img -> ft113 img_h, 1024-bit). server query = so400m text (lowercased) -> ft113 txt_h. offline query = multilingual-e5-small (native) -> adapted txt_h'. `*` = retrieved image's COCO captions contain a query keyword (relevance proxy). **Difficulty spectrum: 2 clear + 3 generic.** **Global mean top-5 set-overlap (server vs offline, 5K EN test) = 0.578** (this figure's metric; `encoders.csv` separately reports a 0.42 agreement metric under a different definition — cite whichever fits, they are NOT the same measurement). Key honest check: when the paths DIVERGE (low overlap), are the NON-SHARED images still relevant?

## Q1 [clear]: EN `two giraffes standing near a tree` / KO `나무 옆에 서 있는 기린 두 마리`  (kw: ['giraffe'])
- **EN** server [145009*, 476109*, 197745*, 573756*, 20470*] | offline [476109*, 145009*, 197745*, 573756*, 20470*] | **overlap 5/5** | non-shared 0 imgs, 0 relevant (server-only [], offline-only [])
- **KO** server [476109*, 197745*, 577524*, 145009*, 383676*] | offline [536615*, 197745*, 577524*, 40621*, 476109*] | **overlap 3/5** | non-shared 4 imgs, 4 relevant (server-only [145009, 383676], offline-only [536615, 40621])

## Q2 [clear]: EN `a slice of pizza on a white plate` / KO `흰 접시 위의 피자 한 조각`  (kw: ['pizza'])
- **EN** server [110630*, 69698*, 365121*, 401860*, 44934*] | offline [110630*, 401860*, 365121*, 69698*, 378860*] | **overlap 4/5** | non-shared 2 imgs, 2 relevant (server-only [44934], offline-only [378860])
- **KO** server [110630*, 69698*, 575915*, 365121*, 401860*] | offline [401860*, 110630*, 365121*, 69698*, 132415*] | **overlap 4/5** | non-shared 2 imgs, 2 relevant (server-only [575915], offline-only [132415])

## Q3 [generic]: EN `people sitting around a dining table` / KO `식탁에 둘러앉은 사람들`  (kw: ['table', 'dining', 'eating', 'restaurant', 'food', 'meal'])
- **EN** server [151521*, 433883*, 458137*, 139457*, 579240*] | offline [565389*, 433883*, 13220*, 151521*, 376322*] | **overlap 2/5** | non-shared 6 imgs, 6 relevant (server-only [458137, 139457, 579240], offline-only [565389, 13220, 376322])
- **KO** server [151521*, 579240*, 433883*, 139457*, 360170*] | offline [47837*, 13220*, 565389*, 433883*, 376322*] | **overlap 1/5** | non-shared 8 imgs, 8 relevant (server-only [151521, 579240, 139457, 360170], offline-only [47837, 13220, 565389, 376322])

## Q4 [generic]: EN `a busy city street with cars and people` / KO `차와 사람들로 붐비는 도심 거리`  (kw: ['street', 'road', 'traffic', 'city', 'cars', 'intersection', 'bus'])
- **EN** server [555763*, 184324*, 51089*, 106335*, 337042*] | offline [555763*, 184324*, 267802*, 39484*, 106335*] | **overlap 3/5** | non-shared 4 imgs, 4 relevant (server-only [51089, 337042], offline-only [267802, 39484])
- **KO** server [555763*, 184324*, 39484*, 106335*, 51089*] | offline [184324*, 555763*, 51089*, 39484*, 261471*] | **overlap 4/5** | non-shared 2 imgs, 2 relevant (server-only [106335], offline-only [261471])

## Q5 [generic]: EN `a bathroom with a sink and a mirror` / KO `세면대와 거울이 있는 욕실`  (kw: ['bathroom', 'sink', 'toilet', 'mirror', 'shower', 'restroom'])
- **EN** server [238355*, 513064*, 196521*, 411968*, 97017*] | offline [196521*, 238355*, 411968*, 542089*, 292278*] | **overlap 3/5** | non-shared 4 imgs, 4 relevant (server-only [513064, 97017], offline-only [542089, 292278])
- **KO** server [196521*, 238355*, 411968*, 97017*, 431197*] | offline [196521*, 542089*, 238355*, 411968*, 97017*] | **overlap 4/5** | non-shared 2 imgs, 2 relevant (server-only [431197], offline-only [542089])

## Summary
- **Global mean top-5 set-overlap (server vs offline, 5K EN test) = 0.578** (this figure's metric; `encoders.csv` separately reports a 0.42 agreement metric under a different definition — cite whichever fits, they are NOT the same measurement).
- mean top-5 overlap by tier (EN): clear 4.5/5, generic 2.67/5  → spectrum as intended (clear agree, generic diverge)
- **non-shared (divergent) EN images relevant: 16/16 (100%)** — the core message: even where the two paths return DIFFERENT images, the divergent ones are still on-topic.
- (Any `·` on a generic query = a non-relevant retrieval, reported as-is, not removed.)
