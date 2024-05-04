import vitaldb

from .relevant_tracks import relevant_track_names



def get_vitaldb_samples(case_id, track_names=relevant_track_names):
    vf = vitaldb.VitalFile(case_id, track_names)
    samples = vf.to_numpy(track_names, 1)
    return samples