import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('data/airlines_delay.csv')  # Replace with your file path

# Plot histograms for numerical features
numerical_features = ['Time', 'Length']
data[numerical_features].hist(bins=30, figsize=(10, 5))
plt.suptitle('Histograms of Numerical Features')
plt.show()

# Plot the distribution of flights across different airlines
plt.figure(figsize=(10, 5))
sns.countplot(data=data, x='Airline')
plt.title('Number of Flights per Airline')
plt.xticks(rotation=45)
plt.show()

# Plot the distribution of flights across different days of the week
plt.figure(figsize=(10, 5))
sns.countplot(data=data, x='DayOfWeek')
plt.title('Number of Flights per Day of the Week')
plt.show()
