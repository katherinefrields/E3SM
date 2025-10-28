import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


def main(shared_path, hybrid_path_h0):
    # Load the baseline physics simulation datasets
    ds_mmf_ref = xr.open_dataset(shared_path + 'h0/1year/mmf_ref/mmf_ref.eam.h0.0003.nc')
    ds_mmf_a = xr.open_dataset(shared_path + 'h0/1year/mmf_a/mmf_a.eam.h0.0003.nc')
    #ds_mmf_b = xr.open_dataset(shared_path + 'h0/1year/mmf_b/mmf_b.eam.h0.0003.nc')
    #ds_mmf_c = xr.open_dataset(shared_path + 'h0/1year/mmf_c/mmf_c.eam.h0.0003.nc')
    ds_nn = xr.open_mfdataset(hybrid_path_h0+'*.eam.h0.0003-*.nc', engine='netcdf4')


    # Ensure the ./figure directory exists before writing
    os.makedirs('./figure', exist_ok=True)

    # Define the path for the output file
    output_path = './figure/output.txt'

    with open(output_path, "w") as f:
        def process_dataset(ds, name):
            f.write(f"\n=== {name.upper()} means ===\n")

            for var_name in ds.data_vars:
                var = ds[var_name]

                # Skip non-numeric variables
                if not np.issubdtype(var.dtype, np.number):
                    continue

                # --- Compute and write mean ---
                mean_val = var.mean().compute().item()
                f.write(f"{name} {var_name}: {mean_val}\n")

                # --- Plot variable over time if possible ---
                if "time" in var.dims:
                    # Average over all non-time dimensions
                    other_dims = [d for d in var.dims if d != "time"]
                    var_over_time = var.mean(dim=other_dims).compute()

                    # Use real time coordinate
                    time = var["time"].values

                    plt.figure(figsize=(8, 4))
                    plt.plot(time, var_over_time, marker='o', linewidth=1)
                    plt.title(f"{name.upper()} - {var_name} over time")
                    plt.xlabel("Time")
                    plt.ylabel(var_name)
                    plt.tight_layout()

                    plot_path = f"./figure/{name}_{var_name}.png"
                    plt.savefig(plot_path, dpi=150)
                    plt.close()

        # Process each dataset
        process_dataset(ds_mmf_ref, "mmf_ref")
        process_dataset(ds_mmf_a, "mmf_a")
        process_dataset(ds_nn, "nn")
        
    '''with open(output_path, "w") as f:
        f.write("=== MMF REF means ===\n")
        for var_name in ds_mmf_ref.data_vars:
            var = ds_mmf_ref[var_name]
            if not np.issubdtype(var.dtype, np.number):
                continue  # skip string or non-numeric variables
            mean_val = var.mean().compute().item()
            f.write(f"mmf ref {var_name}: {mean_val}\n")
            
        f.write("\n=== MMF A means ===\n")
        for var_name in ds_mmf_a.data_vars:
            var = ds_mmf_a[var_name]
            if not np.issubdtype(var.dtype, np.number):
                continue  # skip string or non-numeric variables
            mean_val = var.mean().compute().item()
            f.write(f"mmf a {var_name}: {mean_val}\n")

        f.write("\n=== MMF NN means ===\n")
        for var_name in ds_nn.data_vars:
            var = ds_nn[var_name]
            if not np.issubdtype(var.dtype, np.number):
                continue  # skip string or non-numeric variables
            mean_val = var.mean().compute().item()
            f.write(f"mmf nn {var_name}: {mean_val}\n")'''
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot RMSE for Temperature and Moisture")
    parser.add_argument('shared_path', type=str, help='Path to the shared directory')
    parser.add_argument('hybrid_path_h0', type=str, help='Path to the hybrid h0 directory')
    args = parser.parse_args()

    main(args.shared_path, args.hybrid_path_h0)
