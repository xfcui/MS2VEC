

# Fragment encoding.
averagine_peak_separation_da = 1.0005079
fragment_mz_min = averagine_peak_separation_da * 50.5
fragment_mz_max = 2500.
bin_size = averagine_peak_separation_da

# MS/MS spectrum preprocessing settings.
min_peaks = 10
min_mz_range = 250.
remove_precursor_tolerance = 2  # Da
min_intensity = 0.01
max_peaks_used = 150
scaling = 'sqrt'

charges = 2, 5
