from hmmlearn.hmm import GMMHMM
import scipy.io.wavfile as wvf
from features import mfcc
import os
import pickle
import glob

def read_wavs_trng(emotions, trng_path, pickle_path, use_pickle=False):
    """
        Utility function to read wav files, convert them into MFCC vectors and store in a pickle file
        (Pickle file is useful in case you re-train on the same data changing hyperparameters)
    """
    trng_data = {}
    if use_pickle and os.path.isfile(pickle_path):
        write_pickle = False
        trng_data = pickle.load(open(pickle_path, "rb"))
    else:
        write_pickle = True
        for emo in emotions:
            mfccs = []
            for wavfile in glob.glob(trng_path + '/' + emo + '/*.wav'):
                rate, sig = wvf.read(wavfile)
                mfcc_feat = mfcc(sig, rate)
                mfccs.append(mfcc_feat)
            trng_data[emo] = mfccs
    if write_pickle:
        pickle.dump(trng_data, open(pickle_path, "wb"))
    return trng_data


def obtain_config(emotions, pickle_path, use_pickle=False):
    """
        Utility function to take in parameters to train individual GMMHMMs
    """
    conf = {}
    if not use_pickle:
        for emo in emotions:
            conf[emo] = {}
            print '*' * 50
            print emo
            print '*' * 20
            conf[emo]["n_components"] = int(input("Enter number of components in the GMMHMM: "))
            conf[emo]["n_mix"] = int(input("Enter number of mixtures in the Gaussian Model: "))
        pickle.dump(conf, open(pickle_path, "wb"))
    else:
        conf = pickle.load(open(pickle_path, "rb"))
    return conf


def train_GMMs(emotions, trng_data, GMM_config, pickle_path, use_pickle=False):
    """
        Utility function to train GMMHMMs based on entered confiuration and training data. Returns a dictionary of trained GMMHMM objects and also pickles them for use without training.
    """
    emo_machines = {}
    if not use_pickle:
        for emo in emotions:
            emo_machines[emo] = GMMHMM(n_components=GMM_config[emo]["n_components"], n_mix=GMM_config[emo]["n_mix"])
            if trng_data[emo]:
                # print np.shape(trng_data[emo])
                emo_machines[emo].fit(trng_data[emo])
        pickle.dump(emo_machines, open(pickle_path, "wb"))
    else:
        emo_machines = pickle.load(open(pickle_path, "rb"))
    return emo_machines


def create_training_set(trng_path="./Aditya_Recordings", pickles="./pickles"):
    """
    Description:
        Function that takes in the directory containing training data as raw wavfiles within folders named according to emotion, extracts MFCC feature vectors from them, accepts a configuration for each emotion in terms of number of states for HMM and number of mixtures in the Gaussian Model and then trains a set of GMMHMMs, one for each emotion. All intermediate results are pickled for easier use later.
        This function is invoked only when training = True in predict_emo().

    Params:
        * trng_path (mandatory): Path to the training wav files. Each folder in this path must have the name as emotion and must NOT be empty. If a folder in this path is empty, the emotion will not be considered for classification.

        * pickles (mandatory): Path to store the generated pickle files in. Please keep these constant for the purpose of reuse.

    Return:
        A python dictionary of GMMHMMs that are trained, key values being emotions extracted from folder names.
    """
    emotions = os.listdir(trng_path)
    trng_data = read_wavs_trng(emotions, trng_path, pickle_path=pickles + '/trng_data.pkl')
    GMM_config = obtain_config(emotions, pickle_path=pickles + '/gmm_conf.pkl')
    gmms = train_GMMs(emotions, trng_data, GMM_config, pickle_path=pickles + '/gmmhmm.pkl')
    return gmms


if __name__ == "__main__":
    gmms = create_training_set()
