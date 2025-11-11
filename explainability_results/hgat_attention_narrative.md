### Section 1: Overview

*   **Market:** custom
*   **Number of Stocks:** 97
*   **Checkpoint:** ./checkpoints/ppo\_hgat\_custom\_20251111\_191020.zip

### Section 2: Semantic Attention Interpretation

The HGAT network allocates the highest mean attention to "Positive" relationships (approximately 0.42), suggesting a focus on identifying positively correlated stocks. "Negative" relationships receive the second highest attention (approximately 0.24), likely for hedging or risk management. "Industry" relationships receive a moderate amount of attention (approximately 0.19), indicating some sector-based considerations. "Self" attention has the lowest mean value (approximately 0.15), implying less emphasis on individual stock characteristics in isolation.

### Section 3: Edge-Level Attention Insights

*   **Industry:** MARICO.NS appears as a central node, influencing BRITANNIA.NS, BAJFINANCE.NS, MPHASIS.NS and SRF.NS. HAVELLS.NS influences AXISBANK.NS.
*   **Positive:** CDSL.NS, PVRINOX.NS, AARTIIND.NS, INDIGO.NS and AUROPHARMA.NS are leading positive influencers of OBEROIRLTY.NS.
*   **Negative:** TATACOMM.NS, INDIGO.NS, AARTIIND.NS and GRSE.NS are leading negative influencers of MARICO.NS.

### Section 4: Portfolio Interpretation Summary

The portfolio appears to be constructed with a strong emphasis on inter-stock relationships, particularly positive correlations, and hedging against specific stocks. The concentration of negative attention on MARICO.NS suggests it may be used as a hedge or a diversifier, given its negative influence from multiple stocks. The positive influence on OBEROIRLTY.NS suggests it may be a core holding, with other stocks used to amplify its returns. The industry relationships indicate some sector-based dependencies, particularly with MARICO.NS influencing multiple stocks.