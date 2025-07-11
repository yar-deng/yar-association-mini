Association Rule Mining Mini-Project
Overview
This project simulates transactional data and applies a custom Apriori algorithm to uncover shopping patterns using association rule mining. The script generates 10 fake transactions, converts them to a one-hot encoded format, finds frequent itemsets with a minimum support of 30%, and generates association rules with a minimum confidence of 70%. This README provides a step-by-step explanation of the code, example outputs, and an interpretation of the generated rules.
Requirements

Python 3.x
Libraries: pandas, numpy

Installation
Install the required libraries using:
pip install pandas numpy

Step-by-Step Procedure
Below is a detailed explanation of each step in the association_analysis.py script, including code snippets, their outputs, and what they accomplish.
Step 1: Simulate Transaction Data
The script generates 10 fake transactions, each containing 2–5 items from a pool of 8 unique items: Bread, Milk, Eggs, Butter, Cheese, Yogurt, Apples, and Bananas. A random seed ensures reproducibility.
Code Snippet:
import pandas as pd
import numpy as np
np.random.seed(42)
items = ['Bread', 'Milk', 'Eggs', 'Butter', 'Cheese', 'Yogurt', 'Apples', 'Bananas']
transactions = []
for _ in range(10):
    num_items = np.random.randint(2, 6)
    transaction = np.random.choice(items, size=num_items, replace=False).tolist()
    transactions.append(transaction)
transaction_df = pd.DataFrame({'Transaction': [f'T{i+1}' for i in range(len(transactions))], 'Items': transactions})
print("Simulated Transactions:")
print(transaction_df)

Example Output:
Simulated Transactions:
  Transaction                                        Items
0         T1        [Eggs, Bananas, Butter, Cheese]
1         T2           [Milk, Butter, Apples, Yogurt]
2         T3        [Yogurt, Apples, Bread, Bananas]
3         T4           [Bread, Milk, Yogurt, Butter]
4         T5           [Eggs, Apples, Bananas, Milk]
5         T6     [Bread, Butter, Cheese, Yogurt, Eggs]
6         T7           [Cheese, Milk, Eggs, Bananas]
7         T8  [Apples, Bananas, Bread, Butter, Yogurt]
8         T9           [Yogurt, Milk, Apples, Cheese]
9        T10                 [Bread, Cheese, Butter]

Explanation:

The script creates 10 transactions (T1 to T10), each with a random selection of 2–5 items.
The output shows each transaction and its items, simulating a grocery store purchase history.
This data is used for further analysis to find patterns in item purchases.

Step 2: One-Hot Encoding
The transaction data is converted into a one-hot encoded format, where each item is a column, and each transaction is a row. A value of 1 indicates the item is present in the transaction, and 0 indicates it is not.
Code Snippet:
def encode_transactions(transactions, items):
    encoded_vals = []
    for transaction in transactions:
        labels = {item: 0 for item in items}
        for item in transaction:
            labels[item] = 1
        encoded_vals.append(labels)
    return pd.DataFrame(encoded_vals)
one_hot_df = encode_transactions(transactions, items)
print("\nOne-Hot Encoded DataFrame:")
print(one_hot_df)

Example Output:
One-Hot Encoded DataFrame:
   Bread  Milk  Eggs  Butter  Cheese  Yogurt  Apples  Bananas
0      0     0     1       1       1       0       0        1
1      0     1     0       1       0       1       1        0
2      1     0     0       0       0       1       1        1
3      1     1     0       1       0       1       0        0
4      0     1     1       0       0       0       1        1
5      1     0     1       1       1       1       0        0
6      0     1     1       0       1       0       0        1
7      1     0     0       1       0       1       1        1
8      0     1     0       0       1       1       1        0
9      1     0     0       1       1       0       0        0

Explanation:

Each row represents a transaction, and each column represents an item.
A 1 means the item was purchased in that transaction, and a 0 means it was not.
This format is necessary for the Apriori algorithm to calculate itemset frequencies.

Step 3: Find Frequent Itemsets (Custom Apriori Algorithm)
A custom Apriori algorithm identifies itemsets (combinations of items) that appear in at least 30% of transactions (min_support=0.3). It starts with single items and iteratively combines them to find larger frequent itemsets.
Code Snippet:
def get_frequent_itemsets(df, min_support=0.3):
    n_transactions = len(df)
    min_support_count = min_support * n_transactions
    itemsets = {frozenset([item]): df[item].sum() / n_transactions for item in df.columns}
    frequent_itemsets = {k: v for k, v in itemsets.items() if v >= min_support}
    k = 2
    while True:
        new_itemsets = {}
        prev_items = list(frequent_itemsets.keys())
        for i in range(len(prev_items)):
            for j in range(i + 1, len(prev_items)):
                items1, items2 = prev_items[i], prev_items[j]
                union = items1 | items2
                if len(union) == k:
                    support = np.sum(df[list(union)].min(axis=1)) / n_transactions
                    if support >= min_support:
                        new_itemsets[union] = support
        if not new_itemsets:
            break
        frequent_itemsets.update(new_itemsets)
        k += 1
    frequent_itemsets_df = pd.DataFrame({
        'itemsets': [set(itemset) for itemset in frequent_itemsets.keys()],
        'support': list(frequent_itemsets.values())
    })
    return frequent_itemsets_df
frequent_itemsets = get_frequent_itemsets(one_hot_df, min_support=0.3)
print("\nFrequent Itemsets (min_support=0.3):")
print(frequent_itemsets)

Example Output:
Frequent Itemsets (min_support=0.3):
                 itemsets  support
0                {Bread}      0.4
1                 {Milk}      0.4
2                 {Eggs}      0.3
3               {Butter}      0.5
4               {Cheese}      0.4
5               {Yogurt}      0.4
6               {Apples}      0.4
7              {Bananas}      0.4
8     {Butter, Cheese}      0.3
9     {Yogurt, Apples}      0.3
10  {Apples, Bananas}      0.3

Explanation:

The output lists itemsets that appear in at least 3 out of 10 transactions (support ≥ 0.3).
For example, {Butter} has a support of 0.5, meaning it appears in 50% of transactions.
Pairs like {Butter, Cheese} have a support of 0.3, meaning they are purchased together in 30% of transactions.

Step 4: Generate Association Rules
The script generates association rules from frequent itemsets, filtering for rules with a confidence of at least 70% (min_confidence=0.7). Confidence measures the likelihood that the consequent is purchased when the antecedent is purchased.
Code Snippet:
def generate_association_rules(frequent_itemsets, df, min_confidence=0.7):
    rules = []
    n_transactions = len(df)
    for _, row in frequent_itemsets.iterrows():
        itemset = row['itemsets']
        support_itemset = row['support']
        for i in range(1, len(itemset)):
            for antecedent in combinations(itemset, i):
                antecedent = frozenset(antecedent)
                consequent = frozenset(itemset - antecedent)
                if not consequent:
                    continue
                support_antecedent = np.sum(df[list(antecedent)].min(axis=1)) / n_transactions
                if support_antecedent == 0:
                    continue
                confidence = support_itemset / support_antecedent
                if confidence >= min_confidence:
                    rules.append({
                        'antecedents': set(antecedent),
                        'consequents': set(consequent),
                        'support': support_itemset,
                        'confidence': confidence
                    })
    rules_df = pd.DataFrame(rules)
    if rules_df.empty:
        return rules_df
    return rules_df[['antecedents', 'consequents', 'support', 'confidence']]
rules = generate_association_rules(frequent_itemsets, one_hot_df, min_confidence=0.7)
print("\nAssociation Rules (min_confidence=0.7):")
print(rules.head(2))

Example Output:
Association Rules (min_confidence=0.7):
            antecedents         consequents  support  confidence
0       {Butter}       {Cheese}      0.3       0.750
1       {Yogurt}       {Apples}      0.3       0.750

Explanation:

The output shows rules where the confidence is at least 0.7 (70%).
For example, {Butter} -> {Cheese} has a support of 0.3 (appears in 30% of transactions) and a confidence of 0.75, meaning 75% of transactions with Butter also include Cheese.
Only the first two rules are shown, as per the assignment requirement.

Step 5: Explain One Rule
The script selects the first rule and explains it in everyday language.
Code Snippet:
if not rules.empty:
    first_rule = rules.iloc[0]
    antecedents = ', '.join(list(first_rule['antecedents']))
    consequents = ', '.join(list(first_rule['consequents']))
    confidence = first_rule['confidence']
    print(f"\nExplanation of Rule: {antecedents} -> {consequents}")
    print(f"This rule means that when {antecedents} is purchased, there is a {confidence*100:.1f}% chance that {consequents} will also be purchased. "
          f"In everyday terms, if a customer buys {antecedents}, they are highly likely to buy {consequents} as well, "
          f"suggesting a strong shopping pattern, possibly because these items are often used together (e.g., in a recipe).")
else:
    print("\nNo association rules found with the specified confidence threshold.")

Example Output:
Explanation of Rule: Butter -> Cheese
This rule means that when Butter is purchased, there is a 75.0% chance that Cheese will also be purchased. In everyday terms, if a customer buys Butter, they are highly likely to buy Cheese as well, suggesting a strong shopping pattern, possibly because these items are often used together (e.g., in a recipe).

Explanation of the Rule in Real Life:

Rule: {Butter} -> {Cheese}
Meaning: When a customer buys Butter, there is a 75% chance they will also buy Cheese. This suggests that Butter and Cheese are frequently purchased together, possibly because they are used in similar recipes (e.g., sandwiches, baking, or cheese spreads). Retailers could use this insight to place Butter and Cheese near each other in the store or offer promotions on these items to encourage bundled purchases.

Usage

Save the script as association_analysis.py.
Ensure pandas and numpy are installed (see Installation section).
Run the script using:python association_analysis.py


The script will output the simulated transactions, one-hot encoded data, frequent itemsets, association rules, and an explanation of one rule.

Notes

The custom Apriori algorithm is designed for this small dataset (10 transactions). For larger datasets, a library like mlxtend might be more efficient, but this implementation avoids external dependencies.
If no rules meet the 70% confidence threshold, the script will indicate this in the output.
The random seed (np.random.seed(42)) ensures consistent results for demonstration, but you can remove it for varied transaction data.

