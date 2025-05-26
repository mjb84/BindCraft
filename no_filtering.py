####################################
###################### BindCraft Run
####################################
### Import dependencies
from functions import * # type: ignore

# Check if JAX-capable GPU is available, otherwise exit
check_jax_gpu() # type: ignore

######################################
### parse input paths
parser = argparse.ArgumentParser(description='Script to run BindCraft binder design.') # type: ignore

parser.add_argument('--settings', '-s', type=str, required=True,
                    help='Path to the basic settings.json file. Required.')
parser.add_argument('--filters', '-f', type=str, default='./settings_filters/default_filters.json',
                    help='Path to the filters.json file used to filter design. If not provided, default will be used.')
parser.add_argument('--advanced', '-a', type=str, default='./settings_advanced/default_4stage_multimer.json',
                    help='Path to the advanced.json file with additional design settings. If not provided, default will be used.')

args = parser.parse_args() # type: ignore

# perform checks of input setting files
settings_path, filters_path, advanced_path = perform_input_check(args) # type: ignore

### load settings from JSON
target_settings, advanced_settings, filters = load_json_settings(settings_path, filters_path, advanced_path) # type: ignore

# Get the new setting, default to False if not present
stop_after_mpnn_generation = advanced_settings.get("stop_after_mpnn_generation", False)
# If stopping early, ensure save_mpnn_fasta is true for convenience, or handle saving explicitly later
# For this example, we'll rely on the existing save_mpnn_fasta logic.
# You might want to force it:
# if stop_after_mpnn_generation:
#     advanced_settings["save_mpnn_fasta"] = True


settings_file = os.path.basename(settings_path).split('.')[0] # type: ignore
filters_file = os.path.basename(filters_path).split('.')[0] # type: ignore
advanced_file = os.path.basename(advanced_path).split('.')[0] # type: ignore

### load AF2 model settings
design_models, prediction_models, multimer_validation = load_af2_models(advanced_settings["use_multimer_design"]) # type: ignore

### perform checks on advanced_settings
bindcraft_folder = os.path.dirname(os.path.realpath(__file__)) # type: ignore
advanced_settings = perform_advanced_settings_check(advanced_settings, bindcraft_folder) # type: ignore

### generate directories, design path names can be found within the function
design_paths = generate_directories(target_settings["design_path"]) # type: ignore

### generate dataframes
trajectory_labels, design_labels, final_labels = generate_dataframe_labels() # type: ignore

trajectory_csv = os.path.join(target_settings["design_path"], 'trajectory_stats.csv') # type: ignore
mpnn_csv = os.path.join(target_settings["design_path"], 'mpnn_design_stats.csv') # type: ignore
final_csv = os.path.join(target_settings["design_path"], 'final_design_stats.csv') # type: ignore
failure_csv = os.path.join(target_settings["design_path"], 'failure_csv.csv') # type: ignore

create_dataframe(trajectory_csv, trajectory_labels) # type: ignore
create_dataframe(mpnn_csv, design_labels) # type: ignore
create_dataframe(final_csv, final_labels) # type: ignore
generate_filter_pass_csv(failure_csv, args.filters) # type: ignore

####################################
####################################
####################################
### initialise PyRosetta
pr.init(f'-ignore_unrecognized_res -ignore_zero_occupancy -mute all -holes:dalphaball {advanced_settings["dalphaball_path"]} -corrections::beta_nov16 true -relax:default_repeats 1') # type: ignore
print(f"Running binder design for target {settings_file}")
print(f"Design settings used: {advanced_file}")
print(f"Filtering designs based on {filters_file}")
if stop_after_mpnn_generation:
    print("WARNING: Script will stop after MPNN sequence generation for each trajectory.")
    if not advanced_settings.get("save_mpnn_fasta", False):
        print("INFO: 'save_mpnn_fasta' is false. MPNN sequences will be generated but not saved as FASTA unless explicitly handled.")

####################################
# initialise counters
script_start_time = time.time() # type: ignore
trajectory_n = 1
accepted_designs = 0

### start design loop
while True:
    ### check if we have the target number of binders
    final_designs_reached = check_accepted_designs(design_paths, mpnn_csv, final_labels, final_csv, advanced_settings, target_settings, design_labels) # type: ignore

    if final_designs_reached:
        # stop design loop execution
        break

    ### check if we reached maximum allowed trajectories
    max_trajectories_reached = check_n_trajectories(design_paths, advanced_settings) # type: ignore

    if max_trajectories_reached:
        break

    ### Initialise design
    # measure time to generate design
    trajectory_start_time = time.time() # type: ignore

    # generate random seed to vary designs
    seed = int(np.random.randint(0, high=999999, size=1, dtype=int)[0]) # type: ignore

    # sample binder design length randomly from defined distribution
    samples = np.arange(min(target_settings["lengths"]), max(target_settings["lengths"]) + 1) # type: ignore
    length = np.random.choice(samples) # type: ignore

    # load desired helicity value to sample different secondary structure contents
    helicity_value = load_helicity(advanced_settings) # type: ignore

    # generate design name and check if same trajectory was already run
    design_name = target_settings["binder_name"] + "_l" + str(length) + "_s"+ str(seed)
    trajectory_dirs = ["Trajectory", "Trajectory/Relaxed", "Trajectory/LowConfidence", "Trajectory/Clashing"]
    trajectory_exists = any(os.path.exists(os.path.join(design_paths[trajectory_dir], design_name + ".pdb")) for trajectory_dir in trajectory_dirs) # type: ignore

    if not trajectory_exists:
        print("Starting trajectory: "+design_name)

        ### Begin binder hallucination
        trajectory = binder_hallucination(design_name, target_settings["starting_pdb"], target_settings["chains"], # type: ignore
                                            target_settings["target_hotspot_residues"], length, seed, helicity_value,
                                            design_models, advanced_settings, design_paths, failure_csv)
        # trajectory_metrics = copy_dict(trajectory._tmp["best"]["aux"]["log"]) # type: ignore # Original line
        # Ensure trajectory.aux exists and has the 'log' key, and _tmp and best exist
        if hasattr(trajectory, '_tmp') and "best" in trajectory._tmp and "aux" in trajectory._tmp["best"] and "log" in trajectory._tmp["best"]["aux"]: # type: ignore
            trajectory_metrics = copy_dict(trajectory._tmp["best"]["aux"]["log"]) # type: ignore
        else:
            print(f"Warning: Could not retrieve metrics from trajectory for {design_name}. Skipping detailed metrics for this trajectory.")
            trajectory_metrics = {}


        trajectory_pdb = os.path.join(design_paths["Trajectory"], design_name + ".pdb") # type: ignore

        # round the metrics to two decimal places
        trajectory_metrics = {k: round(v, 2) if isinstance(v, float) else v for k, v in trajectory_metrics.items()}

        # time trajectory
        trajectory_time = time.time() - trajectory_start_time # type: ignore
        trajectory_time_text = f"{'%d hours, %d minutes, %d seconds' % (int(trajectory_time // 3600), int((trajectory_time % 3600) // 60), int(trajectory_time % 60))}"
        print("Starting trajectory took: "+trajectory_time_text)
        print("")

        # Proceed if there is no trajectory termination signal
        # Ensure trajectory.aux and trajectory.aux["log"] exist
        if hasattr(trajectory, 'aux') and "log" in trajectory.aux and trajectory.aux["log"].get("terminate", "Unknown") == "": # type: ignore
            # Relax binder to calculate statistics
            trajectory_relaxed = os.path.join(design_paths["Trajectory/Relaxed"], design_name + ".pdb") # type: ignore
            pr_relax(trajectory_pdb, trajectory_relaxed) # type: ignore

            # define binder chain, placeholder in case multi-chain parsing in ColabDesign gets changed
            binder_chain = "B"

            # Calculate clashes before and after relaxation
            num_clashes_trajectory = calculate_clash_score(trajectory_pdb) # type: ignore
            num_clashes_relaxed = calculate_clash_score(trajectory_relaxed) # type: ignore

            # secondary structure content of starting trajectory binder and interface
            trajectory_alpha, trajectory_beta, trajectory_loops, trajectory_alpha_interface, trajectory_beta_interface, trajectory_loops_interface, trajectory_i_plddt, trajectory_ss_plddt = calc_ss_percentage(trajectory_pdb, advanced_settings, binder_chain) # type: ignore

            # analyze interface scores for relaxed af2 trajectory
            trajectory_interface_scores, trajectory_interface_AA, trajectory_interface_residues = score_interface(trajectory_relaxed, binder_chain) # type: ignore

            # starting binder sequence
            trajectory_sequence = trajectory.get_seq(get_best=True)[0] # type: ignore

            # analyze sequence
            traj_seq_notes = validate_design_sequence(trajectory_sequence, num_clashes_relaxed, advanced_settings) # type: ignore

            # target structure RMSD compared to input PDB
            trajectory_target_rmsd = target_pdb_rmsd(trajectory_pdb, target_settings["starting_pdb"], target_settings["chains"]) # type: ignore

            # save trajectory statistics into CSV
            trajectory_data = [design_name, advanced_settings["design_algorithm"], length, seed, helicity_value, target_settings["target_hotspot_residues"], trajectory_sequence, trajectory_interface_residues,
                                trajectory_metrics.get('plddt', None), trajectory_metrics.get('ptm', None), trajectory_metrics.get('i_ptm', None), trajectory_metrics.get('pae', None), trajectory_metrics.get('i_pae', None),
                                trajectory_i_plddt, trajectory_ss_plddt, num_clashes_trajectory, num_clashes_relaxed, trajectory_interface_scores['binder_score'],
                                trajectory_interface_scores['surface_hydrophobicity'], trajectory_interface_scores['interface_sc'], trajectory_interface_scores['interface_packstat'],
                                trajectory_interface_scores['interface_dG'], trajectory_interface_scores['interface_dSASA'], trajectory_interface_scores['interface_dG_SASA_ratio'],
                                trajectory_interface_scores['interface_hb_bb_sc'], trajectory_interface_scores['interface_hb_sc_sc'],
                                trajectory_interface_scores['interface_hb_bb_bb_longrange'], trajectory_interface_scores['interface_hb_bb_bb_shortrange'],
                                trajectory_alpha, trajectory_beta, trajectory_loops, trajectory_alpha_interface, trajectory_beta_interface, trajectory_loops_interface, traj_seq_notes,
                                trajectory_metrics.get("helix_probability", None), trajectory_target_rmsd, trajectory_time_text, trajectory_metrics.get("terminate", None)]

            add_row_to_csv(trajectory_csv, trajectory_data) # type: ignore

            ### Proceed to mpnn design, if enabled
            if advanced_settings["enable_mpnn"]:
                # mpnn redesign of sequence from trajectory backbone
                mpnn_sequences, mpnn_scores = mpnn_gen_sequence(trajectory, advanced_settings, design_paths, design_name, target_settings["chains"], target_settings["target_hotspot_residues"]) # type: ignore

                if not mpnn_sequences:
                    print("MPNN did not generate any sequences for " + design_name)
                else:
                    print(f"MPNN generated {len(mpnn_sequences)} sequences for {design_name}")

                ################################################################
                ### MODIFICATION START: Check if we should stop after MPNN
                ################################################################
                if stop_after_mpnn_generation:
                    if mpnn_sequences:
                        print(f"Stopping after MPNN generation for {design_name} as requested.")
                        # The mpnn_gen_sequence function already handles saving to FASTA if save_mpnn_fasta is true.
                        # If you wanted to force save here regardless of the flag, you could add:
                        # if not advanced_settings.get("save_mpnn_fasta", False) and mpnn_sequences:
                        #     fasta_path = os.path.join(design_paths["MPNN"], design_name + "_mpnn.fasta")
                        #     with open(fasta_path, "w") as f:
                        #         for i, seq in enumerate(mpnn_sequences):
                        #             f.write(f">{design_name}_mpnn_{i}\n{seq}\n")
                        #     print(f"Saved {len(mpnn_sequences)} MPNN sequences to {fasta_path}")
                    else:
                        print(f"No MPNN sequences generated for {design_name}. Skipping revalidation/filtering (as stop_after_mpnn_generation is true).")

                    # Increment trajectory counter and continue to the next trajectory
                    trajectory_n += 1
                    continue # This skips the rest of the loop for the current trajectory
                ################################################################
                ### MODIFICATION END
                ################################################################

                # initialise mpnn design counters
                mpnn_count = 1
                mpnn_passes = 0

                # predict and score each mpnn designed sequence
                for i in range(len(mpnn_sequences)):
                    if mpnn_passes >= advanced_settings["max_mpnn_sequences"]:
                        break

                    mpnn_design_name = design_name + "_mpnn_" + str(i)
                    mpnn_pdb_filename = mpnn_design_name + ".pdb"
                    mpnn_complex_path = os.path.join(design_paths["MPNN/Complex"], mpnn_pdb_filename) # type: ignore
                    print("Starting design validation for "+mpnn_design_name)

                    # predict protein complex
                    mpnn_complex_metrics, mpnn_complex_pdb = predict_binder_complex(mpnn_sequences[i], target_settings["starting_pdb"], target_settings["chains"], # type: ignore
                                                                                mpnn_design_name, prediction_models, advanced_settings, multimer_validation, design_paths)

                    # predict binder monomer
                    mpnn_monomer_metrics, mpnn_monomer_pdb = predict_binder_alone(mpnn_sequences[i], target_settings["starting_pdb"], target_settings["chains"], # type: ignore
                                                                            mpnn_design_name, prediction_models, advanced_settings, multimer_validation, design_paths)

                    # check if AF2 complex prediction succeeded
                    if mpnn_complex_pdb is not None:
                        # apply initial AF2-based filters to skip non-promising designs
                        mpnn_filters_passed = apply_initial_filters(mpnn_complex_metrics, filters["Initial_filter"]) # type: ignore
                        print("MPNN design passed initial filters: " +str(mpnn_filters_passed))

                        # relax mpnn complex
                        mpnn_relaxed_path = os.path.join(design_paths["MPNN/Relaxed"], mpnn_pdb_filename) # type: ignore
                        pr_relax(mpnn_complex_path, mpnn_relaxed_path) # type: ignore

                        # number of clashes pre and post-relaxation
                        num_clashes_mpnn = calculate_clash_score(mpnn_complex_path) # type: ignore
                        num_clashes_mpnn_relaxed = calculate_clash_score(mpnn_relaxed_path) # type: ignore

                        # analyse mpnn designed interface
                        mpnn_interface_scores, mpnn_interface_AA, mpnn_interface_residues = score_interface(mpnn_relaxed_path, binder_chain) # type: ignore

                        # analyse mpnn sequence
                        mpnn_seq_notes = validate_design_sequence(mpnn_sequences[i], num_clashes_mpnn_relaxed, advanced_settings) # type: ignore

                        # ss content of mpnn binder and interface
                        mpnn_alpha, mpnn_beta, mpnn_loops, mpnn_alpha_interface, mpnn_beta_interface, mpnn_loops_interface, mpnn_i_plddt, mpnn_ss_plddt = calc_ss_percentage(mpnn_relaxed_path, advanced_settings, binder_chain) # type: ignore

                        # target structure RMSD compared to original binding site from input trajectory
                        mpnn_interface_rmsd = interface_pdb_rmsd(trajectory_pdb, mpnn_relaxed_path, target_settings["chains"], binder_chain) # type: ignore

                        # target structure RMSD compared to input PDB
                        mpnn_target_rmsd = target_pdb_rmsd(mpnn_relaxed_path, target_settings["starting_pdb"], target_settings["chains"]) # type: ignore

                        # binder structure RMSD compared to binder monomer
                        mpnn_binder_rmsd = binder_pdb_rmsd(mpnn_relaxed_path, mpnn_monomer_pdb, binder_chain, target_settings["chains"], advanced_settings["remove_binder_monomer"]) # type: ignore

                        mpnn_data = [mpnn_design_name, advanced_settings["design_algorithm"], length, seed, helicity_value, target_settings["target_hotspot_residues"],
                                    mpnn_sequences[i], mpnn_interface_residues, mpnn_scores[i], mpnn_complex_metrics.get("plddt", None), mpnn_complex_metrics.get("ptm", None),
                                    mpnn_complex_metrics.get("i_ptm", None), mpnn_complex_metrics.get("pae", None), mpnn_complex_metrics.get("i_pae", None), mpnn_i_plddt, mpnn_ss_plddt,
                                    num_clashes_mpnn, num_clashes_mpnn_relaxed, mpnn_interface_scores['binder_score'], mpnn_interface_scores['surface_hydrophobicity'],
                                    mpnn_interface_scores['interface_sc'], mpnn_interface_scores['interface_packstat'], mpnn_interface_scores['interface_dG'],
                                    mpnn_interface_scores['interface_dSASA'], mpnn_interface_scores['interface_dG_SASA_ratio'], mpnn_interface_scores['interface_hb_bb_sc'],
                                    mpnn_interface_scores['interface_hb_sc_sc'], mpnn_interface_scores['interface_hb_bb_bb_longrange'], mpnn_interface_scores['interface_hb_bb_bb_shortrange'],
                                    mpnn_alpha, mpnn_beta, mpnn_loops, mpnn_alpha_interface, mpnn_beta_interface, mpnn_loops_interface, mpnn_monomer_metrics.get("plddt", None),
                                    mpnn_monomer_metrics.get("ptm", None), mpnn_monomer_metrics.get("pae", None), mpnn_binder_rmsd, mpnn_interface_rmsd, mpnn_target_rmsd,
                                    mpnn_seq_notes, trajectory_metrics.get("helix_probability", None)]
                        add_row_to_csv(mpnn_csv, mpnn_data) # type: ignore

                        # Apply filters from JSON file
                        mpnn_final_filters_passed = apply_final_filters(mpnn_data, filters["Filters"], design_labels, failure_csv, mpnn_design_name) # type: ignore
                        print("MPNN design passed final filters: " +str(mpnn_final_filters_passed))

                        if mpnn_final_filters_passed:
                            # If filters are passed, copy binder complex and relaxed to "Accepted" folder
                            copy_file(mpnn_complex_path, os.path.join(design_paths["MPNN/Accepted"], mpnn_pdb_filename)) # type: ignore
                            copy_file(mpnn_relaxed_path, os.path.join(design_paths["MPNN/Accepted"], mpnn_design_name + "_relaxed.pdb")) # type: ignore
                            copy_file(mpnn_monomer_pdb, os.path.join(design_paths["MPNN/Accepted"], mpnn_design_name + "_monomer.pdb")) # type: ignore
                            add_row_to_csv(final_csv, mpnn_data) # type: ignore
                            mpnn_passes += 1
                            print("Accepted design: " +mpnn_design_name)
                        else:
                            print("Rejected design: " +mpnn_design_name)

                        # clean-up files if not specified to keep them
                        if advanced_settings["remove_unrelaxed_complex"]:
                            remove_file(mpnn_complex_path) # type: ignore

                        if advanced_settings["remove_binder_monomer"] and mpnn_final_filters_passed is False:
                            remove_file(mpnn_monomer_pdb) # type: ignore

                    else:
                        print("MPNN design prediction failed for: " + mpnn_design_name)
                    print("")
                    mpnn_count += 1

                # clean-up files if not specified to keep them
                if advanced_settings["remove_unrelaxed_trajectory"]:
                    remove_file(trajectory_pdb) # type: ignore

            trajectory_n += 1
            print("Finished trajectory: "+design_name)
            print("------------------------------------------------------------")
            print("")

        else:
            # Trajectory terminated
            print("Trajectory " + design_name + " terminated.")
            trajectory_n += 1
            print("------------------------------------------------------------")
            print("")

    else:
        # Trajectory exists, skipping
        print("Trajectory " + design_name + " already exists, skipping")
        print("")

### print job information
# total time
script_total_time = time.time() - script_start_time # type: ignore
script_total_time_text = f"{'%d hours, %d minutes, %d seconds' % (int(script_total_time // 3600), int((script_total_time % 3600) // 60), int(script_total_time % 60))}"
print("Finished BindCraft job!")
print("Total time: " + script_total_time_text)
