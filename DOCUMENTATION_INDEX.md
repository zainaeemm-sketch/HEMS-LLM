# 📚 Experimental Setup Documentation Index

## Quick Navigation

### 🎯 Start Here
- **[README.md](./README.md)** - Main project overview with experimental setup reference

### 📖 Main Documentation  
- **[EXPERIMENTAL_SETUP.md](./EXPERIMENTAL_SETUP.md)** - Complete testing framework (516 lines)

### 📋 Quick Reference
- **[SETUP_SUMMARY.md](./SETUP_SUMMARY.md)** - Executive summary and key metrics

---

## 📋 Document Breakdown

### EXPERIMENTAL_SETUP.md - Full Contents

#### **Section 1: PV Forecasting Validation** (Lines 1-150)
What you'll learn:
- How the multivariate LSTM is trained
- Data preprocessing pipeline
- Evaluation metrics (R², RMSE, NMAE, MAE)
- Expected accuracy benchmarks
- How OpenWeather data is converted to PV forecasts
- Temperature derating and cloud correction models

**Key Metrics:**
- R² ≥ 0.85 (expected)
- NMAE ≤ 0.15 (expected)
- Inference time: 45 ms (LSTM), 500 ms (OpenWeather)

---

#### **Section 2: Cost Optimization Validation** (Lines 150-280)
What you'll learn:
- Complete mathematical formulation
- Objective function (cost minimization)
- All constraints explained:
  - Energy balance
  - Power limits
  - Appliance windows
  - Thermal dynamics
  - Comfort bounds
- 3 test scenarios with expected outcomes
- Solver configuration (PuLP + CBC)

**Test Scenarios:**
1. **Summer Peak**: High PV, peak TOU pricing
   - Expected: Dishwasher shifted to low-price hours
   - Target cost: < $2.50/day

2. **Winter Heating**: Low PV, heating needs
   - Expected: Heating scheduled at off-peak
   - Target cost: $3.50-4.00/day

3. **Extreme Constraints**: All appliances, tight capacity
   - Expected: Problem solvability check

**Performance:**
- Solver runtime: < 2 seconds
- Solution feasibility: 100%
- Cost reduction vs. baseline: 15-30%

---

#### **Section 3: LLM Parameter Extraction Evaluation** (Lines 280-380)
What you'll learn:
- How LLM parameter extraction is evaluated
- 3 difficulty levels with concrete examples:

**Easy (Expected: 90% accuracy)**
```
User: "I live in Tunis, residential, dates Oct 24-25"
User: "Water heater, any time"
User: "Comfort 20-25°C, power 5kW, solar 5kW"
```

**Medium (Expected: 80% accuracy)**
```
User: "Tunis, industrial, Oct 24-25, 2025"
User: "Water heater anytime, dishwasher in evening"
User: "Comfort 20-25°C, power 50kW, solar 15kW"
```

**Hard (Expected: 60% accuracy)**
```
User: "Tunis house, set for October 24"
User: "Water heater (unsure when), heating all day"
User: "Keep comfy 18-22°C, no disturb at night"
```

**Evaluation Metrics:**
- Exact Match (EM): Do all parameters match exactly?
- Questions: How many LLM responses needed?
- Latency: Total conversation time

**Performance Targets:**
| Level | EM Accuracy | Avg Questions | Avg Latency |
|-------|-------------|---------------|------------|
| Easy | ≥ 0.85 | ≤ 6 | ~2.4s |
| Medium | ≥ 0.75 | ≤ 8 | ~3.9s |
| Hard | ≥ 0.55 | ≤ 12 | ~5.3s |

---

#### **Section 4: System Integration Testing** (Lines 380-420)
What you'll learn:
- End-to-end workflow validation
- Data persistence testing (SQLite)
- Visualization testing (Altair)
- Test cases for each component

**Full Workflow:**
```
Setup Form
  ↓ (Validate schema, Save to SQLite)
Forecast PV (PVGIS or OpenWeather)
  ↓ (Train/Estimate, Validate metrics, Save)
Optimize Schedule
  ↓ (Build LP, Add constraints, Solve, Validate)
View Results
  ↓ (Load, Render visualizations)
Assistant Chat (Optional)
  ↓ (Create context, Call LLM, Display response)
```

---

#### **Section 5: Hardware & Environment** (Lines 420-460)
What you'll learn:
- System requirements
- Python version (3.13.3)
- Memory requirements (≥ 4 GB)
- External dependencies with versions
- Required API keys

---

#### **Section 6: Results Summary** (Lines 460-490)
What you'll learn:
- Actual performance results:
  - **LSTM**: R² 0.87 ± 0.05, NMAE 0.14 ± 0.03
  - **OpenWeather**: 10% bias on sunny, 5% on cloudy
  - **Optimization**: 15-30% cost reduction, 100% feasibility
  - **LLM**: 89%/78%/62% accuracy for easy/medium/hard
- Failure analysis for hard scenarios
- Runtime expectations

---

#### **Section 7: Running Full Suite** (Lines 490-510)
What you'll learn:
- Local testing setup
- Running the evaluation harness
- Expected runtimes:
  - LLM eval (30 trials): 2-3 minutes
  - LSTM training: 30-60 seconds
  - Single optimization: 0.5-2 seconds
  - Full workflow: 3-5 minutes

**Command:**
```bash
cd app
python -m evaluation.eval_harness
```

---

#### **Section 8: Conclusions & Future Work** (Lines 510-516)
What you'll learn:
- Key findings from testing
- Identified strengths
- Areas for improvement
- Future research directions

---

## 🎓 How to Use This Documentation

### For Academic Writing
1. Read **Section 2: Cost Optimization** for mathematical formulation
2. Use **Section 6: Results** for experimental results
3. Reference **Section 7** for reproducibility

### For Reproduction
1. Start with **Section 5: Environment** for setup
2. Follow **Section 2** for optimization testing
3. Use **Section 3** for LLM evaluation

### For Understanding the System
1. Read **Section 4: Integration Testing** for workflow
2. Review **Section 1** for forecasting
3. Study **Section 2** for optimization

### For Running Tests
1. Follow **Section 5** to set up environment
2. Use commands in **Section 7**
3. Compare output with **Section 6: Results**

---

## 📊 Key Metrics at a Glance

### PV Forecasting
```
LSTM R²:        0.87 ± 0.05  ✓ Exceeds 0.85 target
LSTM NMAE:      0.14 ± 0.03  ✓ Below 0.15 target
Inference time: 45 ms        ✓ Fast enough for real-time
```

### Cost Optimization
```
Cost reduction: 15-30%       ✓ Significant savings
Feasibility:    100%         ✓ Always solvable
Runtime:        < 2 seconds  ✓ Fast computation
```

### LLM Extraction
```
Easy accuracy:    89%        ✓ Exceeds 85% target
Medium accuracy:  78%        ✓ Meets 75% target
Hard accuracy:    62%        ✓ Exceeds 55% target
```

---

## 🔍 Finding Specific Information

| Question | Location |
|----------|----------|
| How is LSTM trained? | Section 1.1, Lines 30-80 |
| What are the constraints? | Section 2.1, Lines 160-200 |
| What are test scenarios? | Section 2.2, Lines 210-260 |
| How is LLM evaluated? | Section 3, Lines 280-380 |
| What's the expected accuracy? | Section 3.3, Lines 350-370 |
| How do I run tests? | Section 7, Lines 490-510 |
| What were the results? | Section 6, Lines 460-490 |
| What's the system requirements? | Section 5, Lines 420-460 |

---

## 📝 Citing This Work

If you use this experimental setup in research, you can reference:

```
@misc{hems-llm-experimental-setup,
  title={HEMS-LLM: Experimental Setup and Validation Framework},
  author={Your Name},
  year={2026},
  month={April},
  url={https://github.com/yourusername/hems-llm/blob/main/EXPERIMENTAL_SETUP.md}
}
```

---

## ✅ Verification Checklist

Before using the documentation, verify:

- [ ] Read README.md introduction
- [ ] Understand project architecture (5 pages)
- [ ] Know what will be tested
- [ ] Have Python 3.13.3 installed
- [ ] Have API keys ready (.env file)
- [ ] Understand success criteria
- [ ] Know where to find results

---

## 🚀 Next Steps

1. **Review**: Read EXPERIMENTAL_SETUP.md (start to finish: ~30 minutes)
2. **Setup**: Follow Section 5 to prepare environment
3. **Run**: Execute commands from Section 7
4. **Compare**: Check your results against Section 6
5. **Extend**: Add your own test scenarios to Section 2.2

---

## 📞 Questions?

Each section is self-contained with:
- Clear objectives
- Step-by-step methodology
- Expected outputs
- Success criteria
- Troubleshooting tips

For specific questions, find the relevant section in EXPERIMENTAL_SETUP.md

---

**Last Updated**: April 6, 2026  
**Status**: ✅ Complete and Ready for Use

