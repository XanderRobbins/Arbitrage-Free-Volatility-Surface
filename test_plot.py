import pandas as pd
import matplotlib.pyplot as plt
from vol_surface import VolatilitySurface

df = pd.read_csv('data/spy_options.csv')

surface = VolatilitySurface(S=590, r=0.045)
surface.load_data(df)
surface.compute_ivs()

# Test smile plot
print("Displaying smile plot...")
surface.plot_smile(expiry=df['expiry'].iloc[0])
plt.show()

# Test SVI fitting and surface
print("Fitting SVI and displaying surface...")
surface.fit_svi()
surface.plot_surface_3d(model='svi')
plt.show()

# Plot raw market data (no fitting needed)
print("Displaying market surface...")
surface.plot_surface_3d(model='market')
plt.show()

# Test term structure
print("Displaying term structure...")
surface.term_structure(moneyness=1.0)
plt.show()

surface.summary()
print("\nOK - All plots displayed!")

