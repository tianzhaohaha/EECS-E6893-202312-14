import json
import pandas as pd

with open('relation.json', 'r') as file:
    relations = json.load(file)

comps = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"]

relation_matrix = pd.DataFrame(0, index=comps, columns=comps)

for comp in comps:
    if comp in relations:
        related_comps = relations[comp]
        for related_comp in related_comps:
            if related_comp in relation_matrix.columns:
                relation_matrix.at[comp, related_comp] += 1
                relation_matrix.at[related_comp, comp] += 1

print(relation_matrix)

relation_matrix.to_csv('weight.csv', header=False, index=False)