i# Import the required libraries
import json
import pandas as pd

# Open and load the JSON file containing relations
with open('relation.json', 'r') as file:
    relations = json.load(file)

# Define a list of company symbols
comps = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"]

# Create a DataFrame for the relation matrix with companies as both rows and columns, initialized to 0
relation_matrix = pd.DataFrame(0, index=comps, columns=comps)

# Iterate through each company in the list
for comp in comps:
    # Check if the current company is in the relations dictionary
    if comp in relations:
        # Get the list of related companies from the relations dictionary
        related_comps = relations[comp]
        # Iterate through each related company
        for related_comp in related_comps:
            # Check if the related company is in the columns of the relation matrix
            if related_comp in relation_matrix.columns:
                # Increment the relation count for both (comp, related_comp) and (related_comp, comp) in the matrix
                relation_matrix.at[comp, related_comp] += 1
                relation_matrix.at[related_comp, comp] += 1

# Print the relation matrix to the console
print(relation_matrix)

# Save the relation matrix to a CSV file without headers and indexes
relation_matrix.to_csv('weight.csv', header=False, index=False)
