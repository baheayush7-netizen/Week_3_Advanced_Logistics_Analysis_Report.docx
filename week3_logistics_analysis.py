"""
Week 3: Advanced Data Analysis and Visualization in Logistics
Uses a hypothetical dataset. Run: python week3_logistics_analysis.py
Requires: pandas, numpy, matplotlib
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)
n = 120
regions = np.random.choice(["North", "South", "East", "West"], n)
modes = np.random.choice(["Road", "Rail", "Air", "Sea"], n, p=[0.55, 0.20, 0.15, 0.10])
distance = np.random.randint(50, 1800, n)
volume = np.random.randint(10, 500, n)
speed = {"Road": 55, "Rail": 70, "Air": 550, "Sea": 35}
base_days = {"Road": 2.0, "Rail": 1.4, "Air": 0.7, "Sea": 1.1}
cost_rate = {"Road": 0.085, "Rail": 0.06, "Air": 0.32, "Sea": 0.045}

delivery = np.array([
    distance[i] / speed[modes[i]] * 24 + base_days[modes[i]] + np.random.normal(0, 5)
    for i in range(n)
])
delivery = np.clip(delivery, 1, 120).round(1)
cost = (
    distance * np.array([cost_rate[m] for m in modes])
    + volume * 0.18 + np.random.normal(0, 25, n)
).clip(30).round(2)

data = pd.DataFrame({
    "Shipment_ID": [f"S{i+1:03d}" for i in range(n)],
    "Region": regions, "Transport_Mode": modes,
    "Distance_km": distance, "Shipment_Volume_kg": volume,
    "Delivery_Time_days": delivery, "Transport_Cost_INR": cost
})
data["On_Time"] = (data["Delivery_Time_days"] <= data["Delivery_Time_days"].quantile(0.70)).astype(int)
data.to_csv("week3_hypothetical_logistics_data.csv", index=False)

print("\\nDATA OVERVIEW")
print(data.head())
print(data.info())
print("\\nDESCRIPTIVE STATISTICS")
print(data[["Distance_km", "Shipment_Volume_kg", "Delivery_Time_days", "Transport_Cost_INR"]].describe())
print("\\nMEAN, MEDIAN, MODE")
for col in ["Distance_km", "Shipment_Volume_kg", "Delivery_Time_days", "Transport_Cost_INR"]:
    print(col, "mean=", round(data[col].mean(), 2),
          "median=", round(data[col].median(), 2),
          "mode=", data[col].mode().iloc[0])
print("\\nAVERAGE BY TRANSPORT MODE")
print(data.groupby("Transport_Mode")[["Delivery_Time_days", "Transport_Cost_INR"]].mean().round(2))
print("\\nCORRELATION")
print(data[["Distance_km", "Shipment_Volume_kg", "Delivery_Time_days", "Transport_Cost_INR"]].corr().round(2))

plt.figure(figsize=(8, 5))
plt.hist(data["Delivery_Time_days"], bins=14, edgecolor="black")
plt.title("Distribution of Delivery Time")
plt.xlabel("Delivery time (days)"); plt.ylabel("Shipments")
plt.tight_layout(); plt.savefig("delivery_distribution.png", dpi=150)

plt.figure(figsize=(8, 5))
data.groupby("Transport_Mode")["Transport_Cost_INR"].mean().sort_values().plot(kind="bar")
plt.title("Average Transport Cost by Mode")
plt.xlabel("Transport mode"); plt.ylabel("Average cost (INR)")
plt.tight_layout(); plt.savefig("cost_by_mode.png", dpi=150)

plt.figure(figsize=(8, 5))
data.groupby("Region")["Delivery_Time_days"].mean().sort_values().plot(kind="bar")
plt.title("Average Delivery Time by Region")
plt.xlabel("Region"); plt.ylabel("Average delivery time (days)")
plt.tight_layout(); plt.savefig("delivery_by_region.png", dpi=150)

plt.figure(figsize=(8, 5))
plt.scatter(data["Distance_km"], data["Delivery_Time_days"], alpha=0.65)
plt.title("Distance vs Delivery Time")
plt.xlabel("Distance (km)"); plt.ylabel("Delivery time (days)")
plt.tight_layout(); plt.savefig("distance_vs_delivery.png", dpi=150)

corr = data[["Distance_km", "Shipment_Volume_kg", "Delivery_Time_days", "Transport_Cost_INR"]].corr()
plt.figure(figsize=(7, 5))
plt.imshow(corr, aspect="auto", vmin=-1, vmax=1)
plt.xticks(range(len(corr.columns)), corr.columns, rotation=35, ha="right")
plt.yticks(range(len(corr.columns)), corr.columns)
plt.colorbar(label="Correlation")
plt.title("Correlation Matrix")
plt.tight_layout(); plt.savefig("correlation_matrix.png", dpi=150)
plt.show()
