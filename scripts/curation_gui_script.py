from pathlib import Path
import spikeinterface.full as si

if __name__ == "__main__":
    global_job_kwargs = dict(n_jobs=-1, chunk_duration="10s", progress_bar=True)
    si.set_global_job_kwargs(**global_job_kwargs)
    analyzer_path = Path(r"C:\Users\assad\Desktop\experiment_data\sorting_files\DS45\DS45_20260814\analyzer_clean")
    analyzer = si.load_sorting_analyzer(folder=analyzer_path)
    import spikeinterface.curation as sic
    umg = sic.compute_merge_unit_groups(analyzer)


    phypath =  Path(r"C:\Users\assad\Desktop\experiment_data\sorting_files\DS45\DS45_20260814\phy")
    recpath =Path(r"C:\Users\assad\Desktop\experiment_data\recording_files\DS45\DS45_20260814\preprocess")
    rec = si.load(recpath)
    analyzer.set_temporary_recording(rec)
    si.export_to_phy(sorting_analyzer=analyzer, output_folder=phypath)

