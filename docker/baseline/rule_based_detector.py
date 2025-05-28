# Alternative: Rule-Based Fraud Detection
# rule_based_detector.py
class RuleBasedFraudDetector:
    def __init__(self):
        self.rules = []

    def add_rule(self, name, condition_func, weight=1.0):
        """Add a fraud detection rule"""
        self.rules.append({
            'name': name,
            'condition': condition_func,
            'weight': weight
        })

    def detect_fraud(self, df):
        """Detect fraud using rules"""
        fraud_scores = np.zeros(len(df))
        triggered_rules = [[] for _ in range(len(df))]

        for rule in self.rules:
            rule_triggered = df.apply(rule['condition'], axis=1)
            fraud_scores += rule_triggered * rule['weight']

            for i, triggered in enumerate(rule_triggered):
                if triggered:
                    triggered_rules[i].append(rule['name'])

        return {
            'fraud_scores': fraud_scores,
            'triggered_rules': triggered_rules,
            'is_fraud': fraud_scores > 0.5  # Threshold
        }

# Define common fraud rules
detector = RuleBasedFraudDetector()
detector.add_rule('high_amount', lambda x: x['amount'] > 10000, weight=0.4)
detector.add_rule('weekend_invoice', lambda x: pd.to_datetime(x['invoice_date']).weekday() > 4, weight=0.2)
detector.add_rule('round_number', lambda x: x['amount'] % 100 == 0 and x['amount'] > 1000, weight=0.3)
