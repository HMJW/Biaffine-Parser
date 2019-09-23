from .crf_loss import viterbi
import numpy as np

class Viterbi(object):

    @staticmethod
    def decode_one_inst(arc_scores, label_scores, length):
        arc_scores = arc_scores.detach().cpu().numpy()
        label_scores = label_scores.detach().cpu().numpy()
        # inst, arc_scores, label_scores, max_label_prob_as_arc_prob, viterbi_decode = args
        # for labeled-crf-loss, the default is sum of label prob, already stored in arc_scores
        # if max_label_prob_as_arc_prob:
        #    arc_scores = np.max(label_scores, axis=2)
        candidate_heads = np.array([1] * length * length, dtype=np.int32).reshape(length, length)   
        head_pred = viterbi(length, arc_scores, False, candidate_heads)

        label_score_of_concern = label_scores[
            np.arange(length), head_pred[: length]
        ]
        label_pred = np.argmax(label_score_of_concern, axis=1)
        # Parser.set_predict_result(inst, head_pred, label_pred, label_dict)
        # return inst.eval()
        return head_pred, label_pred
