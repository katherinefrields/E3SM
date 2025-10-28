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
        f.write("=== MMF REF means ===\n")
        for var_name in ds_mmf_ref.data_vars:
            var = ds_mmf_ref[var_name]
            mean_val = var.mean().item()
            f.write(f"ref {var_name}: {mean_val}\n")

        f.write("\n=== MMF A means ===\n")
        for var_name in ds_mmf_a.data_vars:
            var = ds_mmf_a[var_name]
            mean_val = var.mean().item()
            f.write(f"mmf a {var_name}: {mean_val}\n")

        f.write("\n=== NN means ===\n")
        for var_name in ds_nn.data_vars:
            var = ds_nn[var_name]
            mean_val = var.mean().item()
            f.write(f"nn {var_name}: {mean_val}\n")
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot RMSE for Temperature and Moisture")
    parser.add_argument('shared_path', type=str, help='Path to the shared directory')
    parser.add_argument('hybrid_path_h0', type=str, help='Path to the hybrid h0 directory')
    args = parser.parse_args()

    main(args.shared_path, args.hybrid_path_h0)
