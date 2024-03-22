"""Removes ads from episodes of Darknet Diaries"""

from os.path import basename
from sys import argv

import essentia.standard as es
import numpy as np
from pydub import AudioSegment
from tqdm import tqdm


def get_mfcc(filename, sr=44100, frameSize=2048, hopSize=512):
    if filename.endswith(".npy"):
        print(f"Loading MFCCs from {filename}...")
        mfccs = np.load(filename)
        return mfccs, sr, hopSize

    print(f"Loading audio from {filename}...")
    audio = es.MonoLoader(filename=filename, sampleRate=sr)()
    
    # Compute MFCCs
    w = es.Windowing(type='hann')
    spectrum = es.Spectrum()
    mfcc = es.MFCC(numberCoefficients=13)
    mfccs = []
    total_frames = int(np.ceil(len(audio) / hopSize))
    
    for frame in tqdm(
        es.FrameGenerator(audio, frameSize=frameSize, hopSize=hopSize, startFromZero=True),
        total=total_frames,
        desc=f"Extracting MFCCs from {basename(filename)}",
        leave=False,
    ):
        _, mfcc_coeffs = mfcc(spectrum(w(frame)))
        mfccs.append(mfcc_coeffs)
    return np.array(mfccs), sr, hopSize


def find_best_match(
    target_mfccs, podcast_mfccs, sr, hopSize, start_offset=0, distance_threshold=2000.0
):
    min_distance = float('inf')
    best_offset = -1  # Default value indicating no match found
    search_range = range(len(podcast_mfccs) - len(target_mfccs) + 1)
    pbar = tqdm(total=len(search_range), desc="Searching for matches", leave=False)
    
    for offset in search_range:
        distance = np.linalg.norm(podcast_mfccs[offset : offset + len(target_mfccs)] - target_mfccs)
        if distance < min_distance:
            min_distance = distance
            best_offset = offset
        pbar.update()
    pbar.close()

    # Check if the best match found is within the acceptable distance threshold
    if min_distance > distance_threshold:
        print(f"{min_distance} > {distance_threshold}")
        return -1, 0, min_distance  # Indicating no valid match found

    timestamp = (best_offset + start_offset) * hopSize / sr
    return best_offset, timestamp, min_distance


def find_all_ads(podcast_mfccs, ad_start_mfccs, ad_end_mfccs, sr, hopSize):
    ad_segments = []
    current_offset = 0

    while current_offset < len(podcast_mfccs) - len(ad_start_mfccs):
        # Search for ad start from the current offset
        start_best_offset, start_timestamp, start_distance = find_best_match(
            ad_start_mfccs, podcast_mfccs[current_offset:], sr, hopSize, current_offset
        )
        end_search_offset = (
            current_offset + start_best_offset + len(ad_start_mfccs)
        )  # Start searching for the end after the ad start

        # Search for ad end, starting from the end of the detected ad start
        end_best_offset, end_timestamp, end_distance = find_best_match(
            ad_end_mfccs, podcast_mfccs[end_search_offset:], sr, hopSize, end_search_offset
        )

        if start_best_offset == 0 and end_best_offset == 0:  # Break if no new ad is found
            break

        # Adjust the offsets based on the current search position
        adjusted_end_offset = end_search_offset + end_best_offset

        # Update current_offset to the end of the detected ad to search for the next ad
        current_offset = adjusted_end_offset + len(ad_end_mfccs)

        # Save the segment (start, end, s_dist, e_dist)
        ad_segments.append((start_timestamp, end_timestamp + 2, start_distance, end_distance))

    return ad_segments


def cut_out_ads(podcast_filename, ad_segments):
    # Load the podcast episode
    podcast = AudioSegment.from_file(podcast_filename)
    duration_in_ms = len(podcast)

    # We'll accumulate non-ad segments here
    non_ad_segments = []
    last_end_ms = 0

    # Sort ad segments by start time
    ad_segments = sorted(ad_segments, key=lambda x: x[0])

    for start, end, _, _ in ad_segments:
        start_ms = int(start * 1000)
        end_ms = int(end * 1000)

        # Add the non-ad segment before the current ad
        if start_ms > last_end_ms:
            non_ad_segments.append(podcast[last_end_ms:start_ms])

        last_end_ms = end_ms  # Update the end of the last ad

    # Add the remaining part of the podcast after the last ad
    if last_end_ms < duration_in_ms:
        non_ad_segments.append(podcast[last_end_ms:])

    # Concatenate all non-ad segments
    podcast_without_ads = sum(non_ad_segments)

    # Export the result
    output_filename = podcast_filename.replace(".mp3", "_no_ads.mp3")
    podcast_without_ads.export(output_filename, format="mp3")
    return output_filename


def fuck_ads(input_file):
    # Load and extract MFCCs for ad start and ad end segments
    ad_start_mfccs, sr, hop_size = get_mfcc('dd_ad_start_mfccs.npy')
    ad_end_mfccs, _, _ = get_mfcc('dd_ad_end_mfccs.npy')
    podcast_mfccs, _, _ = get_mfcc(input_file)

    # Find all ads
    ad_segments = find_all_ads(podcast_mfccs, ad_start_mfccs, ad_end_mfccs, sr, hop_size)
    for i, (start, end, s_distance, e_distance) in enumerate(ad_segments, 1):
        print(
            f"Ad {i}: Start at {start} seconds, ends at {end} seconds (s_dis: {s_distance} - e_dis: {e_distance})"
        )

    print("Removing ads...")
    output_filename = cut_out_ads(input_file, ad_segments)
    print(f"Podcast without ads saved as: {output_filename}")


if __name__ == "__main__":
    if len(argv) != 2:
        print(f"Usage: {argv[0]} episode.mp3")
        exit(1)
    fuck_ads(argv[1])
