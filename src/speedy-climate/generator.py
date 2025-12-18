import pandas as pd
import numpy as np
from scipy.stats import qmc

import subprocess
import os
import shutil
import time

import concurrent.futures

from constants import *

HELIOS_path = ''

HELIOS_path = '/data/pt426/HELIOS'
CUDA_path = '/data/pt426/cuda/cuda12/bin'
CUDA_LD_path = '/data/pt426/cuda/cuda12/lib64'
CUDA_DYLD_path = '/data/pt426/cuda/cuda12/lib'

def run_HELIOS(name: str, instellation: float, spectral_type: str, R_planet: float, M_planet: float, P_surface: float, P_CO2: float, P_H2O: float, albedo: float, recirculation_factor: float, verbose: bool=False) -> dict[str, object]:

    g_surface = (G * M_planet) / (R_planet ** 2)

    R_star = SPECTRAL_TYPE_DATA[spectral_type]['Radius']
    T_star = SPECTRAL_TYPE_DATA[spectral_type]['Temperature']

    orbital_distance = np.sqrt((((R_star * R_SUN) ** 2) * STEFAN_BOLTZMANN * (T_star ** 4)) / (instellation)) / AU

    species_data = {
        'species' : ['H2O', 'CO2'],
        'absorbing' : ['yes', 'yes'],
        'scattering' : ['yes', 'yes'],
        'mixing_ratio' : [P_H2O / P_surface, P_CO2 / P_surface]
    }
        
    species_df = pd.DataFrame(species_data)
    species_df.to_csv(f'{HELIOS_path}/input/species_{name}.dat', sep='\t', index=False)

    command = [f'.venv/bin/python', 'helios.py']

    parameters = [
        '-name', name,
        '-boa_pressure', f'{P_surface / 10}',
        '-f_factor', f'{recirculation_factor}',
        # '-stellar_zenith_angle', f'{zenith_angle}',
        '-surface_albedo', f'{albedo}',
        '-surface_gravity', f'{g_surface * 100}',
        '-radius_planet', f'{R_planet / R_JUPITER}',
        '-path_to_species_file', f'./input/species_{name}.dat', 
        '-radius_star', f'{R_star}',
        '-temperature_star', f'{T_star}',
        '-orbital_distance', f'{orbital_distance}'
    ]

    env = os.environ.copy()
    env["PATH"] = CUDA_path + ":" + env["PATH"]
    env["LD_LIBRARY_PATH"] = CUDA_LD_path + ":" + env.get("LD_LIBRARY_PATH", "")
    env["DYLD_LIBRARY_PATH"] = CUDA_DYLD_path + ":" + env.get("LD_LIBRARY_PATH", "")

    if verbose:
        subprocess.run(command + parameters, cwd=HELIOS_path, env=env) 
    else:
        subprocess.run(command + parameters, cwd=HELIOS_path, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    try:
        atm_df = pd.read_table(f'{HELIOS_path}/output/{name}/{name}_tp.dat', sep='\s+', skiprows=1)

        # P = np.array(atm_df['press.[10^-6bar]']) * 10
        T = np.array(atm_df['temp.[K]'])
        # z = np.array(atm_df['altitude[cm]']) / 100

        T_surface = float(T[0])

        result_dict = {
            "Run_ID": name,
            # "Zenith_Angle": zenith_angle,
            "Instellation (W/m^2)": instellation,
            "Spectral_Type": spectral_type,
            "R_Planet (m)": R_planet,
            "M_Planet (kg)": M_planet,
            "P_Surface (Pa)": P_surface,
            "P_CO2 (Pa)": P_CO2,
            "P_H2O (Pa)": P_H2O,
            "Albedo": albedo,
            "Recirculation_Factor": recirculation_factor,
            "Surface_Temp (K)": T_surface,
            "Status": "Success"
        }

    except Exception as e :
        
        result_dict = {
            "Run_ID": name,
            # "Zenith_Angle": zenith_angle,
            "Instellation (W/m^2)": instellation,
            "Spectral_Type": spectral_type,
            "R_Planet (m)": R_planet,
            "M_Planet (kg)": M_planet,
            "P_Surface (Pa)": P_surface,
            "P_CO2 (Pa)": P_CO2,
            "P_H2O (Pa)": P_H2O,
            "Albedo": albedo,
            "Recirculation_Factor": recirculation_factor,
            "Surface_Temp (K)": -1,
            "Status": str(e)
        }

    shutil.rmtree(f'{HELIOS_path}/output/{name}', ignore_errors=True)
    
    return result_dict

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
	
