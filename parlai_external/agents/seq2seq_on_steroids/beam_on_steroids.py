try:
    import torch
except ImportError as e:
    raise ImportError('Need to install Pytorch: go to pytorch.org')
from collections import namedtuple, Counter
import editdistance.bycython as ed
import math
from operator import attrgetter
from parlai.core.utils import NEAR_INF

class Beam(object):
    def __init__(self, beam_size, min_length=3, padding_token=0, bos_token=1, eos_token=2, min_n_best=3, cuda='cpu', block_ngram=0):
        """
        Generic beam class. It keeps information about beam_size hypothesis.
        :param beam_size: number of hypothesis in the beam
        :param min_length: minimum length of the predicted sequence
        :param padding_token: Set to 0 as usual in ParlAI
        :param bos_token: Set to 1 as usual in ParlAI
        :param eos_token: Set to 2 as usual in ParlAI
        :param min_n_best: Beam will not be done unless this amount of finished hypothesis (with EOS) is done
        :param cuda: What device to use for computations
        """
        self.beam_size = beam_size
        self.min_length = min_length
        self.eos = eos_token
        self.bos = bos_token
        self.pad = padding_token
        self.device = cuda
        self.scores = torch.Tensor(self.beam_size).float().zero_().to(
            self.device)  # recent score for each hypo in the beam
        self.all_scores = [torch.Tensor([0.0] * beam_size).to(self.device)]  # self.scores values per each time step
        self.bookkeep = []  # backtracking id to hypothesis at previous time step
        self.outputs = [
            torch.Tensor(self.beam_size).long().fill_(bos_token).to(self.device)]  # output tokens at each time step
        self.finished = []  # keeps tuples (score, time_step, hyp_id)
        self.HypothesisTail = namedtuple('HypothesisTail', ['timestep', 'hypid', 'score', 'tokenid'])
        self.eos_top = False
        self.eos_top_ts = None
        self.n_best_counter = 0
        self.min_n_best = min_n_best
        self.block_ngram = block_ngram
        self.partial_hyps = [[0] for i in range(beam_size)]
        #import ipdb; ipdb.set_trace()

    @staticmethod
    def find_ngrams(input_list, n):
        return list(zip(*[input_list[i:] for i in range(n)]))

    def get_output_from_current_step(self):
        return self.outputs[-1]

    def get_backtrack_from_current_step(self):
        return self.bookkeep[-1]

    def hamming_dist(self, l1, l2):
        assert len(l1) == len(l2)
        res = 0
        for i in range(len(l1)):
            if l1[i] != l2[i]:
                res += 1
        return res

    def advance(self, softmax_probs, partials=None, min_hamming_dist=1):
        if partials is None:
            partials = []
        else:
            partials = [h[1:] for h in partials]
        voc_size = softmax_probs.size(-1)
        current_length = len(self.all_scores) - 1
        if current_length < self.min_length:
            # penalize all eos probs to make it decode longer
            for hyp_id in range(softmax_probs.size(0)):
                softmax_probs[hyp_id][self.eos] = -NEAR_INF
        if len(self.bookkeep) == 0:
            #  the first step
            beam_scores = softmax_probs[
                0].to(self.device)  # we take only the first hypo into account since all hypos are the same initially
        else:
            #  we need to sum up hypo scores and current softmax scores before topk
            beam_scores = softmax_probs.to(self.device) + self.scores.unsqueeze(1).expand_as(softmax_probs)  # [beam_size, voc_size]

            for i in range(self.outputs[-1].size(0)):
                current_hypo = self.partial_hyps[i][1:]
                current_ngrams = []
                for ng in range(self.block_ngram):
                    ngrams = Beam.find_ngrams(current_hypo, ng)
                    if len(ngrams) > 0:
                        current_ngrams.extend(ngrams)
                counted_ngrams = Counter(current_ngrams)
                if any(v > 1 for k,v in counted_ngrams.items()):
                    # block this hypothesis hard
                    beam_scores[i] = -NEAR_INF

                # block based on hamm distance
                current_length = len(self.outputs) - 1  # excluding starting NULL
                #  this is used for parallel blocking beam
                if len(partials) > 0:
                    for neighbor_cand in partials:
                        if len(neighbor_cand) > current_length:
                            dist = self.hamming_dist(neighbor_cand[:current_length], current_hypo)
                            if dist < 1:
                                last_tok = neighbor_cand[-1]
                                beam_scores[i][last_tok] = -NEAR_INF

                #  if previous timestep output hypo of this beam token had eos - we penalize those word probs to never be chosen
                if self.outputs[-1][i] == self.eos:
                    beam_scores[i] = -NEAR_INF  # beam_scores[i] is voc_size array for i-th hypo

        flatten_beam_scores = beam_scores.view(-1)  # [beam_size * voc_size]
        with torch.no_grad():
            best_scores, best_idxs = torch.topk(flatten_beam_scores, self.beam_size, dim=-1)

        self.scores = best_scores
        self.all_scores.append(self.scores)
        hyp_ids = best_idxs / voc_size  # get the backtracking hypothesis id as a multiple of full voc_sizes
        tok_ids = best_idxs % voc_size  # get the actual word id from residual of the same division

        self.outputs.append(tok_ids)
        self.bookkeep.append(hyp_ids)
        self.partial_hyps = [self.partial_hyps[hyp_ids[i]] + [tok_ids[i].item()] for i in range(self.beam_size)]

        for i in range(self.beam_size):
            if self.scores[i] < -NEAR_INF/2:
                self.outputs[-1][i] = self.eos
                self.scores[i] = -1000.0


        #  check new hypos for eos label, if we have some, add to finished
        for hypid in range(self.beam_size):
            if self.outputs[-1][hypid] == self.eos:
                #  this is finished hypo, adding to finished
                eostail = self.HypothesisTail(timestep=len(self.outputs) - 1, hypid=hypid, score=self.scores[hypid],
                                              tokenid=self.eos)
                self.finished.append(eostail)
                self.n_best_counter += 1

        if self.outputs[-1][0] == self.eos:
            self.eos_top = True
            if self.eos_top_ts is None:
                self.eos_top_ts = len(self.outputs) - 1

    def done(self):
        return self.eos_top and self.n_best_counter >= self.min_n_best

    def get_top_hyp(self):
        """
        Helper function to get single best hypothesis
        :return: hypothesis sequence and the final score
        """
        top_hypothesis_tail = self.get_rescored_finished(n_best=1)[0]
        return self.get_hyp_from_finished(top_hypothesis_tail), top_hypothesis_tail.score

    def get_hyp_from_finished(self, hypothesis_tail):
        """
        Extract hypothesis ending with EOS at timestep with hyp_id
        :param timestep: timestep with range up to len(self.outputs)-1
        :param hyp_id: id with range up to beam_size-1
        :return: hypothesis sequence
        """
        assert self.outputs[hypothesis_tail.timestep][hypothesis_tail.hypid] == self.eos
        assert hypothesis_tail.tokenid == self.eos
        hyp_idx = []
        endback = hypothesis_tail.hypid
        for i in range(hypothesis_tail.timestep, -1, -1):
            hyp_idx.append(self.HypothesisTail(timestep=i, hypid=endback, score=self.all_scores[i][endback],
                                               tokenid=self.outputs[i][endback]))
            endback = self.bookkeep[i - 1][endback]

        return hyp_idx

    def get_partial_hyp_from_tail(self, ts, hypid):
        hypothesis_tail = self.HypothesisTail(timestep=ts, hypid=torch.Tensor([hypid]).long(), score=self.all_scores[ts][hypid], tokenid=self.outputs[ts][hypid])
        hyp_idx = []
        endback = hypothesis_tail.hypid
        for i in range(hypothesis_tail.timestep, -1, -1):
            hyp_idx.append(self.HypothesisTail(timestep=i, hypid=endback, score=self.all_scores[i][endback],
                                               tokenid=self.outputs[i][endback]))
            endback = self.bookkeep[i - 1][endback]

        return hyp_idx


    @staticmethod
    def get_pretty_hypothesis(list_of_hypotails):
        hypothesis = []
        for i in list_of_hypotails:
            hypothesis.append(i.tokenid)

        hypothesis = torch.stack(list(reversed(hypothesis)))

        return hypothesis

    def get_rescored_finished(self, n_best=None):
        """

        :param n_best: how many n best hypothesis to return
        :return: list with hypothesis
        """
        rescored_finished = []
        for finished_item in self.finished:
            current_length = finished_item.timestep + 1
            length_penalty = math.pow((1 + current_length) / 6, 0.65)  # this is from Google NMT paper
            rescored_finished.append(self.HypothesisTail(timestep=finished_item.timestep, hypid=finished_item.hypid,
                                                         score=finished_item.score / length_penalty,
                                                         tokenid=finished_item.tokenid))

        srted = sorted(rescored_finished, key=attrgetter('score'), reverse=True)

        if n_best is not None:
            srted = srted[:n_best]

        return srted

    def check_finished(self):
        """
        this function checks if self.finished is empty and adds hyptail
        in that case (this will be suboptimal hypothesis since
        the model did not get any EOS)
        :return: None
        """
        if len(self.finished) == 0:
            # we change output because we want outputs to have this eos to pass assert in L102, it is ok since empty self.finished means junk prediction anyway
            self.outputs[-1][0] = self.eos
            hyptail = self.HypothesisTail(timestep=len(self.outputs) - 1, hypid=0, score=self.all_scores[-1][0],
                                              tokenid=self.outputs[-1][0])

            self.finished.append(hyptail)

    def get_beam_dot(self, dictionary=None, n_best=None):
        """
        Creates pydot graph representation of the beam
        :param outputs: self.outputs from the beam
        :param dictionary: tok 2 word dict to save words in the tree nodes
        :return: pydot graph
        """
        try:
            import pydot
        except ImportError:
            print("Please install pydot package to dump beam visualization")

        graph = pydot.Dot(graph_type='digraph')
        outputs = [i.tolist() for i in self.outputs]
        bookkeep = [i.tolist() for i in self.bookkeep]
        all_scores = [i.tolist() for i in self.all_scores]
        if n_best is None:
            n_best = int(self.beam_size / 2)

        # get top nbest hyp
        top_hyp_idx_n_best = []
        n_best_colors = ['aquamarine', 'chocolate1', 'deepskyblue', 'green2', 'tan']
        end_color = 'yellow'
        sorted_finished = self.get_rescored_finished(n_best=n_best)
        for hyptail in sorted_finished:
            top_hyp_idx_n_best.append(self.get_hyp_from_finished(
                hyptail))  # do not include EOS since it has rescored score not from original self.all_scores, we color EOS with black

        # create nodes
        for tstep, lis in enumerate(outputs):
            for hypid, token in enumerate(lis):
                if tstep == 0:
                    hypid = 0  # collapse all __NULL__ nodes
                node_tail = self.HypothesisTail(timestep=tstep, hypid=hypid, score=all_scores[tstep][hypid],
                                                tokenid=token)
                color = 'white'
                rank = None
                for i, hypseq in enumerate(top_hyp_idx_n_best):
                    if node_tail in hypseq:
                        if n_best <= 5:  # color nodes only if <=5
                            color = n_best_colors[i]
                        rank = i
                        break
                label = "<{}".format(
                    dictionary.vec2txt([token]) if dictionary is not None else token) + " : " + "{:.{prec}f}>".format(
                    all_scores[tstep][hypid], prec=3)
                graph.add_node(pydot.Node(node_tail.__repr__(), label=label, fillcolor=color, style='filled',
                                          xlabel='{}'.format(rank) if rank is not None else ''))
        # create edges
        for revtstep, lis in reversed(list(enumerate(bookkeep))):
            for i, prev_id in enumerate(lis):
                from_node = graph.get_node('"{}"'.format(
                    self.HypothesisTail(timestep=revtstep, hypid=prev_id, score=all_scores[revtstep][prev_id],
                                        tokenid=outputs[revtstep][prev_id]).__repr__()))[0]
                to_node = graph.get_node('"{}"'.format(
                    self.HypothesisTail(timestep=revtstep + 1, hypid=i, score=all_scores[revtstep + 1][i],
                                        tokenid=outputs[revtstep + 1][i]).__repr__()))[0]
                newedge = pydot.Edge(from_node.get_name(), to_node.get_name())
                graph.add_edge(newedge)

        return graph
