import scipy.io.wavfile as wvf
from features import mfcc
import sys
import pickle
import heapq


def test_emo(test_file, gmms):
    """
        NOTE: Use only after training.
        Test a given file and predict an emotion for it.
    """
    rate, sig = wvf.read(test_file)
    mfcc_feat = mfcc(sig, rate)
    pred = {}
    for emo in gmms:
        pred[emo] = gmms[emo].score(mfcc_feat)
    return emotions_nbest(pred, 2), pred


def emotions_nbest(d, n):
    """
        Utility function to return n best predictions for emotion.
    """
    return heapq.nlargest(n, d, key=lambda k: d[k])


def predict_emo(test_file, pickle_path = "./pickles"):
    """
        Description:
            Based on training or testing mode, takes appropriate path to predict emotions for input wav file.

        Params:
            * test_file (mandatory): Wav file for which emotion should be predicted.
            * pickle_path: Default value is same directory as the python file. Path to store pickle files for use later.
            * trng_path: Default is a folder called training in the enclosing directory. Folder containing training data.
            * training: Default is False. if made True, will start the training procedure before testing, else will used previously trained model to test.

        Return:
            A list of predicted emotion and next best predicted emotion.
    """
    gmms = pickle.load(open(pickle_path + "/gmmhmm.pkl", "rb"))
    predicted = test_emo(test_file, gmms)
    return predicted


if __name__ == "__main__":
    if (sys.argv[1]):
        predicted, probs = predict_emo(sys.argv[1])
        print "EMOTION PREDICTED: %s" % predicted[0]
        print "Next Best: %s" % predicted[1]
        print "All: {}".format(probs)
