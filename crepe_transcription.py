from MonoNote import MonoNote
from hmmlearn import hmm
import numpy as np
import crepe
import math
import matplotlib.pyplot as plt


def hz_to_midi_integer(hz):
    if hz == 0:
        return 0
    else:
        return int(round(pretty_midi.hz_to_note_number(hz)))


def quantify_f0(f0, state_file=None):
    f0_quantized = np.array([hz_to_midi_integer(f) for f in f0])
    if state_file:
        state_data = np.load(state_file)
        f0_quantized[np.where(state_data == 3)] = 0
    return f0_quantized


def predict_voicing(confidence):
    """
    Find the Viterbi path for voiced versus unvoiced frames.
    Parameters
    ----------
    confidence : np.ndarray [shape=(N,)]
        voicing confidence array, i.e. the confidence in the presence of
        a pitch
    Returns
    -------
    voicing_states : np.ndarray [shape=(N,)]
        HMM predictions for each frames state, 0 if unvoiced, 1 if
        voiced
    """

    # uniform prior on the voicing confidence
    starting = np.array([0.5, 0.5])

    # transition probabilities inducing continuous voicing state
    transition = np.array([[0.99, 0.01], [0.01, 0.99]])

    # mean and variance for unvoiced and voiced states
    means = np.array([[0.0], [1.0]])
    variances = np.array([[0.25], [0.25]])

    # fix the model parameters because we are not optimizing the model
    model = hmm.GaussianHMM(n_components=2)
    model.startprob_, model.covars_, model.transmat_, model.means_, model.n_features = \
        starting, variances, transition, means, 1

    # find the Viterbi path
    voicing_states = model.predict(confidence.reshape(-1, 1), [len(confidence)])

    return np.array(voicing_states)


if __name__ == '__main__':
    wav_path = r'vibrato_quantize_error.wav'
    sample_rate = 16000
    wav, sr = librosa.load('example.wav', sr=sample_rate)

    time, frequency, confidence, activation = crepe.predict(wav, sr=16000, viterbi=True, step_size=4)
    f0 = frequency
    is_voiced = predict_voicing(confidence)
    frequency_unvoiced = frequency * is_voiced
    f0 = frequency_unvoiced

    mn = MonoNote()
    smoothedPitch = []
    for iFrame in range(len(f0)):
        temp = []
        if f0[iFrame] > 0:  # negative value: silence
            tempPitch = 12 * math.log(f0[iFrame] / 440.0) / math.log(2.0) + 69
            temp += [[tempPitch, 0.9]]
        smoothedPitch += [temp]

    f0_quantized = quantify_f0(f0)

    print('Start HMM')
    mnOut = mn.process(smoothedPitch)

    hmm_note = [mn.pitch for mn in mnOut]
    hmm_note_quantized = np.around(hmm_note)

    plt.figure()
    plt.plot(f0)
    plt.show()

    plt.figure()
    plt.plot(f0_quantized)
    plt.show()

    plt.figure()
    plt.plot(hmm_note * is_voiced)
    plt.show()

    plt.figure()
    plt.plot(hmm_note_quantized * is_voiced)
    plt.show()

    print(mnOut)
