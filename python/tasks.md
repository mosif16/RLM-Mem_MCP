# RLM Tool Improvement Tasks

## Phase 1: Quick Wins

- [x] 1.1 Default quality scans to OFF in run_ios_scan and run_security_scan
- [x] 1.2 Default min_confidence to MEDIUM instead of LOW
- [x] 1.3 Add ios-strict scan mode (no quality metrics)
- [x] 1.4 Add better summary output with severity grouping

## Phase 2: Context-Aware Analysis

- [x] 2.1 Add guard detection for force unwraps (check if preceded by if let/guard let)
- [ ] 2.2 Add function boundary detection (future)
- [ ] 2.3 Add scope-aware nil checking (future)

## Phase 3: Deep iOS Analysis

- [x] 3.1 Add Keychain security scanner
- [x] 3.2 Add CloudKit sync pattern scanner

## Testing

- [x] 4.1 Test all scanners run without regex errors (94 patterns OK)
- [x] 4.2 Test min_confidence filtering works correctly
- [x] 4.3 Test guard detection reduces false positives
- [x] 4.4 Run full iOS scan on test Swift code

## Completed: 2026-01-18

### Changes Made:
1. **Quality scans OFF by default** - `include_quality=False` parameter added
2. **min_confidence=MEDIUM by default** - Reduces noise significantly
3. **ios-strict mode** - HIGH confidence only, no quality checks
4. **Better summary output** - Severity grouping with emojis, collapsible low-priority
5. **Guard detection** - Force unwraps preceded by if-let/guard-let marked LOW confidence
6. **Keychain scanner** - SecItemAdd/Update checks, biometric fallback, hardcoded secrets
7. **CloudKit sync scanner** - Operation handlers, change token, conflict handling
8. **String literal filtering** - Text("Hello!") no longer flagged as force unwrap
