import numpy as np

from torch.utils.data import DataLoader
from utils.save_samples import save_generated_samples_singlegroup


def classify_cond_gen_samples(exp, labels, cond_samples):
    labels = np.reshape(labels, (labels.shape[0], len(exp.labels)))
    clfs = exp.clfs
    eval_labels = dict()
    for l, l_key in enumerate(exp.labels):
        eval_labels[l_key] = dict()
    for key in clfs.keys():
        if key in cond_samples.keys():
            mod_cond_gen = cond_samples[key]
            mod_clf = clfs[key]
            attr_hat = mod_clf(mod_cond_gen)
            for l, label_str in enumerate(exp.labels):
                score = exp.eval_label(attr_hat.cpu().data.numpy(), labels,
                                       index=l)
                # alt.
                # score = exp.eval_label(attr_hat, labels, index=l)
                eval_labels[label_str][key] = score
        else:
            print(str(key) + 'not existing in cond_gen_samples')
    return eval_labels


def calculate_coherence(exp, samples):
    clfs = exp.clfs
    mods = exp.modalities
    # TODO: make work for num samples NOT EQUAL to batch_size
    c_labels = dict()
    for j, l_key in enumerate(exp.labels):
        pred_mods = np.zeros((len(mods.keys()), exp.config.batch_size))
        for k, m_key in enumerate(mods.keys()):
            mod = mods[m_key]
            clf_mod = clfs[mod.name]
            samples_mod = samples[mod.name]
            attr_mod = clf_mod(samples_mod)
            output_prob_mod = attr_mod.cpu().data.numpy()
            pred_mod = np.argmax(output_prob_mod, axis=1).astype(int)
            pred_mods[k, :] = pred_mod
        coh_mods = np.all(pred_mods == pred_mods[0, :], axis=0)
        coherence = np.sum(coh_mods.astype(int)) / float(exp.config.batch_size)
        c_labels[l_key] = coherence
    return c_labels


def test_generation(exp, dm):
    mods = exp.modalities
    mm_vae = exp.mm_vae
    subsets = exp.subsets

    gen_perf = {'cond': dict(),
                'random': dict()}
    for j, l_key in enumerate(exp.labels):
        gen_perf['cond'][l_key] = dict()
        for k, s_key in enumerate(subsets.keys()):
            if s_key != '':
                gen_perf['cond'][l_key][s_key] = dict()
                for m, m_key in enumerate(mods.keys()):
                    gen_perf['cond'][l_key][s_key][m_key] = []
        gen_perf['random'][l_key] = []

    # d_loader = dm.test_dataloader()
    d_loader = DataLoader(dm.dataset_test,
                          batch_size=exp.config.batch_size,
                          shuffle=True,
                          num_workers=4, drop_last=True)

    for iteration, batch in enumerate(d_loader):
        batch_d = batch[0]
        batch_l = batch[1]
        rand_gen = mm_vae.generate()
        coherence_random = calculate_coherence(exp, rand_gen)
        for j, l_key in enumerate(exp.labels):
            gen_perf['random'][l_key].append(coherence_random[l_key])

        if (exp.config.batch_size * iteration) < exp.config.evaluation['num_samples_fid']:
            save_generated_samples_singlegroup(exp, iteration,
                                               'random',
                                               rand_gen)
            save_generated_samples_singlegroup(exp, iteration,
                                               'real',
                                               batch_d)
        for k, m_key in enumerate(batch_d.keys()):  # or as in repo enum(mods.keys())
            batch_d[m_key] = batch_d[m_key].to(exp.device)

        inferred = mm_vae.inference(batch_d)
        lr_subsets = inferred['subsets']
        cg = mm_vae.cond_generation(lr_subsets)
        for k, s_key in enumerate(cg.keys()):
            clf_cg = classify_cond_gen_samples(exp,
                                               batch_l,
                                               cg[s_key])
            for j, l_key in enumerate(exp.labels):
                for m, m_key in enumerate(mods.keys()):
                    gen_perf['cond'][l_key][s_key][m_key].append(clf_cg[l_key][m_key])
            if (exp.config.batch_size * iteration) < exp.config.evaluation['num_samples_fid']:
                save_generated_samples_singlegroup(exp, iteration,
                                                   s_key,
                                                   cg[s_key])
    for j, l_key in enumerate(exp.labels):
        for k, s_key in enumerate(subsets.keys()):
            if s_key != '':
                for l, m_key in enumerate(mods.keys()):
                    perf = exp.mean_eval_metric(gen_perf['cond'][l_key][s_key][m_key])
                    gen_perf['cond'][l_key][s_key][m_key] = perf
        gen_perf['random'][l_key] = exp.mean_eval_metric(gen_perf['random'][l_key])
    return gen_perf
