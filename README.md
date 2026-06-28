# abr-nmr-phase0
## Metatron Dynamics, Inc.

Phase 0 and Phase 0b simulations for the Relational NMR Receiver Array.

### Structure

```
sim/
    declaration.py       Phase 0 domain declaration
    signal.py            Phase 0 spin echo signal model
    noise.py             Phase 0 thermal noise and SNR model
    declaration_0b.py    Phase 0b multi-tissue domain declaration
    signal_0b.py         Phase 0b per-element signal model
    operators_0b.py      ABRCE operators for 8-element array

experiments/
    run_phase0.py        Phase 0 entry point + parameter sweep
    run_phase0b.py       Phase 0b entry point + boundary detection
    outputs/             Generated figures and reports

tests/
    test_phase0.py       Verifier tests for Phase 0
    test_phase0b.py      Verifier tests for Phase 0b
```

### Run

```powershell
pip install -r requirements.txt
python experiments\run_phase0.py
python experiments\run_phase0b.py
```

### Bounded over D. No claim beyond D.
