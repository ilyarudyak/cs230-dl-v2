for previous_start, previous_end in previous_segments:
    if segment_start <= previous_end and segment_end >= previous_start:
        overlap = True