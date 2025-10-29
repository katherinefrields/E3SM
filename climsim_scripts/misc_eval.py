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

    p_interface = ds_mmf_ref.hyai.values[np.newaxis,:,np.newaxis]*ds_mmf_ref.P0.values + ds_mmf_ref.hybi.values[np.newaxis,:,np.newaxis]*ds_mmf_ref.PS.values[:,np.newaxis,:]
    dp = p_interface[:,1:61,:] - p_interface[:,0:60,:]
    area = ds_mmf_ref.area
    area_weight = area.values[np.newaxis,np.newaxis,:]
    total_weight = dp*area_weight
    
    # Ensure the ./figure directory exists before writing
    os.makedirs('./figure', exist_ok=True)

    # Define the path for the output file
    output_path = './figure/output.txt'
    log_output_path = './figure/log.txt'

    with open(output_path, "w") as f:
        with open(log_output_path, "w") as l:
            
            def process_dataset(ds, name):
                f.write(f"\n=== {name.upper()} means ===\n")
                # --- Compute and write mean ---
                #mean_val = var.mean().compute().item()
                #f.write(f"{name} {var_name}: {mean_val}\n")

                #l.write(f'ds var name {ds[var_name].shape}\n')
                #l.write(f'ds ref var name {ds_mmf_ref[var_name].shape}\n')
                
                monthly_ref_mean = ds_mmf_ref.mean(dim=['lev', 'ncol'])
                l.write(f'monthly ref mean shape {monthly_ref_mean.dims}\n')
                
                monthly_nn_mean = ds.mean(dim=['lev', 'ncol'])
                
                year_data = monthly_nn_mean - monthly_ref_mean
                #averaged_year_data = year_data.mean(axis=(1,2))
                months = np.arange(1, 13)
                
                l.write(f'year variable data {year_data.data_vars}\n')
                
                for var_name in ['CLDICE,' 'CLDLIQ', 'T', 'Q', 'PS', 'U', 'V']:
                    var = year_data[var_name]
                    l.write(f'{ds.dims}\n')
                    
                    # Skip non-numeric variables
                    if not np.issubdtype(var.dtype, np.number):
                        continue

                    
                    #total_weight_sliced = total_weight[:12, :, :]
                    
                    #weighted_year_data = year_data.mean(axis=(1,2)) * total_weight_sliced
                    
                    
                    # --- Plot variable over time if possible ---
                    l.write(f'variable data for {var_name}: {year_data[var_name]}\n')
                    
                    plt.figure(figsize=(8, 4))
                    plt.plot(months, year_data[var_name], marker='o', linewidth=1)
                    plt.title(f"{name.upper()} - {var_name} over time")
                    plt.xlabel("Time")
                    plt.ylabel(var_name)
                    plt.tight_layout()

                    plot_path = f"./figure/{name}_{var_name}.png"
                    plt.savefig(plot_path, dpi=150)
                    plt.close()

            # Process each dataset
            #process_dataset(ds_mmf_ref, "mmf_ref")
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
