# Modeling inputs and processing

Each modeled record is one sample-condition combination from the curated literature reference. The public source map joins sample identifiers to DOI/source keys and publication years. These identifiers are shared by both CV schemes and the chronological manifest. The source-aware split treats the DOI/source key as the group.

## Processed input schema

The author-provided CSV should contain the 32 inputs in the feature dictionary, the target lgkp, sample_id, source_key and Year. The chronological analysis additionally uses alloy_id. The fitting-metadata analysis consumes n_points and R2 under the fitting convention of its source. Optional fit-window fields are t_min and t_max. The primary input is the final processed modeling table.

Composition is in atomic percent, temperature is in °C, and heat-treatment and oxidation exposures are in hours. The target is the base-10 logarithm of the rate constant after conversion to g² cm⁻⁴ s⁻¹. The preprocessing entry point preserves supplied heat-treatment flags and recreates a missing flag only when stage temperature and time are supplied with the original missingness convention. It fills missing stage values and elemental fractions with zero. Oxidation temperature, exposure duration and lgkp must be supplied. It does not infer or change kinetic targets, fitting quality, or raw curve points.

The ordered outer fold files are reconstructed from published sample IDs, rather than by selecting new folds. Source-aware inner validation uses group-wise four-fold splits. Random inner validation uses four-fold shuffled splits with seed 42 plus the outer fold number. Inner memberships were recovered under the pinned scikit-learn version from the saved ordered outer training folds and these recorded splitters.

## Kinetic curation and fitting metadata

The manuscript describes domain review of the literature curves and separate fitting of a parabolic regime. Any reanalysis of the curve-selection procedure needs the author's original records and fit windows. The controlled preprocessing code starts from curated modeling inputs. Additional details on the provisional fit-metadata results are in the fitting-metadata folder.

The reported year and the date used for the independent chronological cutoff are separate fields. The latter follows the publication convention and explicit online-date overrides in the chronological source manifest. The within-fold historical-to-current experiment uses its own publication-year restriction as specified in the manuscript.

