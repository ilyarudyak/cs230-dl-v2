import numpy as np
from pydub import AudioSegment
import random
import sys
import io
import os
import glob
import IPython
from td_utils import *


def get_random_time_segment(segment_ms):
    """
    Gets a random time segment of duration segment_ms in a 10,000 ms audio clip.

    Arguments:
    segment_ms -- the duration of the audio clip in ms ("ms" stands for "milliseconds")

    Returns:
    segment_time -- a tuple of (segment_start, segment_end) in ms
    """

    # Make sure segment doesn't run past the 10sec background
    segment_start = np.random.randint(low=0, high=10000 - segment_ms)
    segment_end = segment_start + segment_ms - 1

    return segment_start, segment_end


def is_overlapping(segment, previous_segments):
    """
    Checks if the time of a segment overlaps with the times of existing segments.

    Arguments:
    segment_time -- a tuple of (segment_start, segment_end) for the new segment
    previous_segments -- a list of tuples of (segment_start, segment_end) for the existing segments

    Returns:
    True if the time segment overlaps with any of the existing segments, False otherwise
    """

    segment_start, segment_end = segment

    ### START CODE HERE ### (≈ 4 line)
    # Step 1: Initialize overlap as a "False" flag. (≈ 1 line)
    overlap = False

    # Step 2: loop over the previous_segments start and end times.
    # Compare start/end times and set the flag to True if there is an overlap (≈ 3 lines)
    for previous_segment in previous_segments:
        if is_two_segments_overlapping(segment, previous_segment):
            overlap = True
            break
    ### END CODE HERE ###

    return overlap


def is_two_segments_overlapping(segment, previos_segment):
    s, e = segment
    ps, pe = previos_segment
    return (s <= ps <= e <= pe) or \
           (ps <= s <= pe <= e) or \
           (ps <= s <= e <= pe) or \
           (s <= ps <= pe <= e)


def insert_audio_clip(background, audio_clip, previous_segments):
    """
    Insert a new audio segment over the background noise at a random time step, ensuring that the
    audio segment does not overlap with existing segments.

    Arguments:
    background -- a 10 second background audio recording.
    audio_clip -- the audio clip to be inserted/overlaid.
    previous_segments -- times where audio segments have already been placed

    Returns:
    new_background -- the updated background audio
    """

    # Get the duration of the audio clip in ms
    segment_ms = len(audio_clip)

    ### START CODE HERE ###
    # Step 1: Use one of the helper functions to pick a random time segment onto which to insert
    # the new audio clip. (≈ 1 line)
    segment_time = 0

    # Step 2: Check if the new segment_time overlaps with one of the previous_segments. If so, keep
    # picking new segment_time at random until it doesn't overlap. (≈ 2 lines)
    segment_time = get_random_time_segment(segment_ms)
    # print(segment_time)
    while is_overlapping(segment_time, previous_segments):
        segment_time = get_random_time_segment(segment_ms)
        # print(segment_time)

    # Step 3: Add the new segment_time to the list of previous_segments (≈ 1 line)
    previous_segments.append(segment_time)

    ### END CODE HERE ###

    # Step 4: Superpose audio segment and background
    new_background = background.overlay(audio_clip, position=segment_time[0])

    return new_background, segment_time


def insert_ones(y, segment_end_ms):
    """
    Update the label vector y. The labels of the 50 output steps strictly after the end of the segment
    should be set to 1. By strictly we mean that the label of segment_end_y should be 0 while, the
    50 follow inf labels should be ones.


    Arguments:
    y -- numpy array of shape (1, Ty), the labels of the training example
    segment_end_ms -- the end time of the segment in ms

    Returns:
    y -- updated labels
    """

    # duration of the background (in terms of spectrogram time-steps)
    Ty = 1375
    segment_end_y = int(segment_end_ms * Ty / 10000.0)

    # Add 1 to the correct index in the background label (y)
    ### START CODE HERE ### (≈ 3 lines)
    ones_start = segment_end_y + 1
    ones_end = min(ones_start + 50, Ty)
    print(f'ones_start={ones_start} ones_end={ones_end}')
    y[0, ones_start:ones_end] = 1.0
    ### END CODE HERE ###

    return y


if __name__ == '__main__':
    activates, negatives, backgrounds = load_raw_audio()
    np.random.seed(5)
    audio_clip, segment_time = insert_audio_clip(backgrounds[0], activates[0], [(3790, 4400)])
    print("Segment Time: ", segment_time)
