import pandas as pd
import numpy as np
from scipy.stats import qmc

import time

import concurrent.futures

from climate import run_HELIOS
from constants import *


HELIOS_path = '/data/pt426/HELIOS'

def run_batch_simulation(inputs, output_csv_name="helios_results.csv"):
    
    start_time = time.time()
    results_list = []

    print('Beginning HELIOS model sweep:')

    total_models = len(inputs)
    count = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        
        futures = []

        for args in inputs:
            futures.append(executor.submit(run_HELIOS, *args))

        for future in concurrent.futures.as_completed(futures):
            
            result = future.result()
            results_list.append(result)

            count += 1

            if result["Status"] == "Success":
                print(f"Finished {result['Run_ID']}: {result['Surface_Temp (K)']} K [{count}/{total_models}]")
            else:
                print(f"Failed {result['Run_ID']}: {result.get('Status')} [{count}/{total_models}]")

    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")

    if results_list:
        df = pd.DataFrame(results_list)
        df.to_csv(output_csv_name, index=False)
        print(f"\nResults successfully saved to {output_csv_name}")
    else:
        print("No results were generated.")

def generate_input_parameters(n_samples):

    param_bounds = {
        'instellation': (0.1 * SOLAR_CONSTANT, 1.5 * SOLAR_CONSTANT),
        'log_p_co2': (-1, 5),
        'log_p_h2o': (-1, 5),
        'albedo': (0, 1.0),
    }

    n_dimensions = len(param_bounds)
    l_bounds = [b[0] for b in param_bounds.values()]
    u_bounds = [b[1] for b in param_bounds.values()]

    sampler = qmc.LatinHypercube(d=n_dimensions)

    samples_unit_cube = sampler.random(n=n_samples)
    samples_scaled = qmc.scale(samples_unit_cube, l_bounds, u_bounds)

    inputs = []

    for i, sample in enumerate(samples_scaled):

        # Format: (name, instellation, spec_type, Rp, Mp, P_surf, P_CO2, P_H2O, alb, recirc)
        sample_tuple = (f'run_{i}', float(sample[0]), 'G2', 1.0 * R_EARTH, 1.0 * M_EARTH, 1e5, 10 ** float(sample[1]), 10 ** float(sample[2]), float(sample[3]), 0.25)
        inputs.append(sample_tuple)

    return inputs
        
if __name__ == '__main__':

    # Format: (name, instellation, spec_type, Rp, Mp, P_surf, P_CO2, P_H2O, alb, recirc)
    inputs_example = [
        ("Sim_01", 200, "G2", 1.0 * R_EARTH, 1.0 * M_EARTH, 1e5, 30, 100, 0.3, 0.25),
        ("Sim_02", 400, "G2", 1.0 * R_EARTH, 1.0 * M_EARTH, 1e5, 30, 100, 0.3, 0.25),
        ("Sim_03", 600, "G2", 1.0 * R_EARTH, 1.0 * M_EARTH, 1e5, 30, 100, 0.3, 0.25),
        ("Sim_04", 800, "G2", 1.0 * R_EARTH, 1.0 * M_EARTH, 1e5, 30, 100, 0.3, 0.25),
        #("Sim_05", 1000, "G2", 1.0 * R_EARTH, 1.0 * M_EARTH, 1e5, 30, 100, 0.3, 0.25),
        #("Sim_06", 1200, "G2", 1.0 * R_EARTH, 1.0 * M_EARTH, 1e5, 30, 100, 0.3, 0.25),
        #("Sim_07", 1400, "G2", 1.0 * R_EARTH, 1.0 * M_EARTH, 1e5, 30, 100, 0.3, 0.25),
        #("Sim_08", 1600, "G2", 1.0 * R_EARTH, 1.0 * M_EARTH, 1e5, 30, 100, 0.3, 0.25),
    ]

    run_HELIOS('test', 1.5 * SOLAR_CONSTANT, 'G2', R_EARTH, M_EARTH, EARTH_ATM, 0.9 * EARTH_ATM, 0.9 * EARTH_ATM, 0.0, 0.25, verbose=True)

    # run_batch_simulation(inputs)

    # generate_input_parameters(10)
	
