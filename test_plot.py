import matplotlib
matplotlib.use('Agg')
import pandas as pd
from vol_surface import VolatilitySurface

df = pd.read_csv('data/spy_options.csv')

surface = VolatilitySurface(S=590, r=0.045)
surface.load_data(df)
surface.compute_ivs()

# Test smile plot
import matplotlib.pyplot as plt
surface.plot_smile(expiry=df['expiry'].iloc[0])
import os
import tempfile

temp_dir = tempfile.gettempdir()

plt.savefig(os.path.join(temp_dir, 'smile.png'))
plt.close()
print("OK - Smile plot saved")

# Test SVI fitting and surface
surface.fit_svi()
surface.plot_surface_3d(model='svi')
plt.savefig(os.path.join(temp_dir, 'svi_surface.png'))
plt.close()
print("OK - SVI surface plot saved")

# Plot raw market data (no fitting needed)
surface.plot_surface_3d(model='market')
plt.savefig(os.path.join(temp_dir, 'market_surface.png'))
plt.close()
print("OK - Market surface plot saved")

# Test term structure
surface.term_structure(moneyness=1.0)
plt.savefig(os.path.join(temp_dir, 'term_structure.png'))
plt.close()
print("OK - Term structure plot saved")

surface.summary()
print("\nOK - All smoke tests passed!")

