import numpy as np
import pandas as pd
import nltk
import spacy
import re
import string
from sklearn.preprocessing import OneHotEncoder

# from nltk.corpus import words
# from nltk.corpus import wordnet as wn
# from nltk.stem import PorterStemmer

PHO_LIST = [  # http://www.speech.cs.cmu.edu/cgi-bin/cmudict (37 out of 39)
    "AA",  # odd
    "AE",  # at
    "AH",  # hut
    "AO",  # ought
    "AW",  # cow
    "AY",  # hide
    "B",  # be
    "CH",  # cheese
    "D",  # dee
    "DH",  # thee
    "EH",  # Ed
    "ER",  # hurt
    "EY",  # ate
    "F",  # fee
    "G",  # green
    "HH",  # he
    "IH",  # it
    "IY",  # eat
    "JH",  # gee
    "K",  # key
    "L",  # lee
    "M",  # me
    "N",  # knee
    "NG",  # ping
    "OW",  # oat
    "OY",  # toy
    "P",  # pee
    "R",  # read
    "S",  # sea
    "SH",  # she
    "T",  # tea
    "TH",  # theta
    "UH",  # hood
    "UW",  # two
    "V",  # vee
    "W",  # we
    "Y",  # yield
    "Z",  # zee
    "ZH",  # seizure
]

POA_LIST = [  # (9)
    "vowel",
    "bilabial",
    "post-alveolar",
    "alveolar",
    "interdental",
    "velar",
    "glottal",
    "labiodental",
    "palatal",
]

MOA_LIST = [  # (9)
    "vowel",
    "stop",
    "affricate",
    "fricative",
    "flap",
    "lateral-liquid",
    "nasal",
    "retroflex-liquid",
    "glide",
]

VOI_LIST = ["vowel", "voiced", "voiceless"]

DEP_LIST_FROM_SPACY = [  # https://spacy.io/models/en (45)
    "ROOT",
    "acl",
    "acomp",
    "advcl",
    "advmod",
    "agent",
    "amod",
    "appos",
    "attr",
    "aux",
    "auxpass",
    "case",
    "cc",
    "ccomp",
    "compound",
    "conj",
    "csubj",
    "csubjpass",
    "dative",
    "dep",
    "det",
    "dobj",
    "expl",
    "intj",
    "mark",
    "meta",
    "neg",
    "nmod",
    "npadvmod",
    "nsubj",
    "nsubjpass",
    "nummod",
    "oprd",
    "parataxis",
    "pcomp",
    "pobj",
    "poss",
    "preconj",
    "predet",
    "prep",
    "prt",
    "punct",
    "quantmod",
    "relcl",
    "xcomp",
]

POS_LIST_FROM_SPACY = [  # https://universaldependencies.org/u/pos/ (17)
    "ADJ",
    "ADP",
    "ADV",
    "AUX",  # unique
    "CCONJ",
    "DET",
    "INTJ",  # unique
    "NOUN",
    "NUM",
    "PART",
    "PRON",
    "PROPN",  # unique
    "PUNCT",
    "SCONJ",  # unique
    "SYM",  # unique
    "VERB",
    "X",
]


POS_LIST_FROM_NTLK = [  # https://www.nltk.org/book/ch05.html (11 out of 12)
    "ADJ",
    "ADP",
    "ADV",
    "CONJ",
    "DET",
    "NOUN",
    "NUM",
    "PRT",
    "PRON",
    "VERB",
    ".",
    "X",
]

ENGLISH_PREFIXES = [
    "anti",  # e.g. anti-goverment, anti-racist, anti-war
    "auto",  # e.g. autobiography, automobile
    "de",  # e.g. de-classify, decontaminate, demotivate
    "dis",  # e.g. disagree, displeasure, disqualify
    "down",  # e.g. downgrade, downhearted
    "extra",  # e.g. extraordinary, extraterrestrial
    "hyper",  # e.g. hyperactive, hypertension
    "il",  # e.g. illegal 1
    "im",  # e.g. impossible 1
    "in",  # e.g. insecure 1
    "ir",  # e.g. irregular 1
    "inter",  # e.g. interactive, international
    "mega",  # e.g. megabyte, mega-deal, megaton
    "mid",  # e.g. midday, midnight, mid-October
    "mis",  # e.g. misaligned, mislead, misspelt
    "non",  # e.g. non-payment, non-smoking
    "over",  # e.g. overcook, overcharge, overrate
    "out",  # e.g. outdo, out-perform, outrun
    "post",  # e.g. post-election, post-warn
    "pre",  # e.g. prehistoric, pre-war
    "pro",  # e.g. pro-communist, pro-democracy
    "re",  # e.g. reconsider, redo, rewrite
    "semi",  # e.g. semicircle, semi-retired
    "sub",  # e.g. submarine, sub-Saharan
    "super",  # e.g. super-hero, supermodel
    "tele",  # e.g. television, telephathic
    "trans",  # e.g. transatlantic, transfer
    "ultra",  # e.g. ultra-compact, ultrasound
    "un",  # e.g. under-cook, underestimate
    "up",  # e.g. upgrade, uphill
]

NOUN_SUFFIXES = [
    "age",  # e.g. baggage, village, postage
    "al",  # e.g. arrival, burial, deferral
    "ance",  # e.g. reliance 1
    "ence",  # e.g. defence, insistence 1
    "dom",  # e.g. boredom, freedom, kingdom
    "ee",  # e.g. employee, payee, trainee
    "er",  # e.g. driver, writer 2
    "or",  # e.g. director 2
    "hood",  # e.g. brotherhood, childhood, neighbourhood
    "ism",  # e.g. capitalism, Marxism, socialism (philosophies)
    "ist",  # e.g. capitalist, Marxist, socialist (followers of philosophies)
    "ity",  # e.g. brutality, equality 3
    "ty",  # e.g. cruelty 3
    "ment",  # e.g. amazement, disappointment, parliament
    "ness",  # e.g. happiness, kindness, usefulness
    "ry",  # e.g. entry, ministry, robbery
    "ship",  # e.g. friendship, membership, workmanship
    "sion",  # e.g. expression 4
    "tion",  # e.g. population 4
    "xion",  # e.g. complexion 4
]

VERB_SUFFIXES = [
    "ate",  # e.g. complicate, dominate, irritate
    "en",  # e.g. harden, soften, shorten
    "ify",  # e.g. beautify, clarify, identify
    "ise",  # e.g. economise, realise (-ise is most common in British English) 1
    "ize",  # e.g. industrialize (-ize is most common in American English) 1
]

ADJ_SUFFIXES = [
    "able",  # e.g.  drinkable, portable 1
    "ible",  # e.g. flexible 1
    "al",  # e.g. brutal, formal, postal
    "en",  # e.g. broken, golden, wooden
    "ese",  # e.g. Chinese, Japanese, Vietnamese
    "ful",  # e.g. forgetful, helpful, useful
    "i",  # e.g. Iraqi, Pakistani, Yemeni
    "ic",  # e.g. classic, Islamic, poetic
    "ish",  # e.g. British, childish, Spanish
    "ive",  # e.g. active, passive, productive
    "ian",  # e.g. Canadian, Malaysian, Peruvian
    "less",  # e.g. homeless, hopeless, useless
    "ly",  # e.g. daily, monthly, yearly
    "ous",  # e.g. cautious, famous, nervous
    "y",  # e.g. cloudy, rainy, windy
]

ADV_SUFFIXES = [
    "ly",  # e.g. calmly, easily, quickly
    "ward",  # e.g. downward, homeward, upward 1
    "wards",  # e.g. downwards, homeward(s), upwards 1
    "wise",  # e.g. anti-clockwise, clockwise, edgewise
]

ENGLISH_SUFFIXES = NOUN_SUFFIXES + VERB_SUFFIXES + ADJ_SUFFIXES + ADV_SUFFIXES


def get_prefix_suffix(df):
    prefix_list = ["ADV", "ADJ", "VERB", "NOUN"]

    def get_prefix(word):
        for prefix in sorted(ENGLISH_PREFIXES, key=len, reverse=True):
            if word.lower().startswith(prefix):
                return prefix
        return ""

    def get_suffix_word(word, suffix_list):
        for suffix in sorted(suffix_list, key=len, reverse=True):
            if word.lower().endswith(suffix):
                return suffix
        return ""

    def get_suffix(df, pos, suffix_list):
        df.loc[df.part_of_speech == pos, "suffix"] = df.loc[
            df.part_of_speech == pos, "word_without_punctuation"
        ].apply(get_suffix_word, args=(suffix_list,))
        return df

    df.loc[df.part_of_speech.isin(prefix_list), "prefix"] = df.loc[
        df.part_of_speech.isin(prefix_list), "word_without_punctuation"
    ].apply(get_prefix)
    df = get_suffix(df, "NOUN", NOUN_SUFFIXES)
    df = get_suffix(df, "VERB", VERB_SUFFIXES)
    df = get_suffix(df, "ADJ", ADJ_SUFFIXES)
    df = get_suffix(df, "ADV", ADV_SUFFIXES)

    df.prefix = df.prefix.fillna("").astype(str)
    df.suffix = df.suffix.fillna("").astype(str)

    return df


def get_semantic_info(df):
    sp = spacy.load("en_core_web_sm")

    def get_spacy_stuff(df):
        sentence = " ".join(df.word_without_punctuation)
        original_words = [
            word.replace(" ", "") for word in df.word_without_punctuation.tolist()
        ]
        tokens = []  # tokens
        deps = []  # dependency
        poss = []  # part of speech
        stops = []  # is_stop
        shapes = []  # shape

        doc = sp(sentence)
        word_idx = 0
        spacy_token = []
        dep = []
        pos = []
        stop = []
        shape = []
        new_word = ""
        for token in doc:  # loop through tokens
            new_word = new_word + token.text  # append text
            new_word = new_word.replace(" ", "")  # strip space
            if new_word in original_words[word_idx]:
                spacy_token.append(token.text)
                dep.append(token.dep_)
                pos.append(token.pos_)
                stop.append(token.is_stop)
                shape.append(token.shape_)
            if len(new_word) == len(original_words[word_idx]):
                assert (
                    new_word == original_words[word_idx]
                ), f"{new_word}, {original_words[word_idx]}, {word_idx}"
                # append to list
                tokens.append(spacy_token)
                deps.append(dep)
                poss.append(pos)
                stops.append(stop)
                shapes.append(shape)

                # reset
                spacy_token = []
                dep = []
                pos = []
                stop = []
                shape = []
                new_word = ""
                word_idx += 1  # move to next word
        assert len(deps) == len(poss) == len(df)
        df["token"] = tokens
        df["dep"] = deps
        df["pos"] = poss
        df["is_stop"] = stops
        df["shape"] = shapes
        return df

    df = df.groupby(df.sentence_idx).apply(get_spacy_stuff)
    df = df.explode(["token", "dep", "pos", "is_stop", "shape"], ignore_index=False)
    df["token_idx"] = (df.groupby(["word", "onset", "offset"]).cumcount()).astype(int)
    df = df[df.pos != "SPACE"]

    return df


def add_pos_nltk(df):
    # Get Part of Speech
    words_orig, part_of_speech = zip(
        *nltk.pos_tag(df.word_without_punctuation, tagset="universal")
    )
    df = df.assign(part_of_speech=part_of_speech)

    pos_list = [
        "ADJ",  # adjective	new, good, high, special, big, local
        "ADP",  # adposition	on, of, at, with, by, into, under
        "ADV",  # adverb	really, already, still, early, now
        "CONJ",  # conjunction	and, or, but, if, while, although
        "DET",  # determiner, article	the, a, some, most, every, no, which
        "NOUN",  # noun	year, home, costs, time, Africa
        "NUM",  # numeral	twenty-four, fourth, 1991, 14:24
        "PRT",  # particle	at, on, out, over per, that, up, with
        "PRON",  # pronoun	he, their, her, its, my, I, us
        "VERB",  # verb	is, say, told, given, playing, would
        ".",  # punctuation marks	. , ; !
        "X",  # other	ersatz, esprit, dunno, gr8, univeristy
    ]

    # Get function content
    function_content_dict = {
        "ADP": "function",
        "CONJ": "function",
        "DET": "function",
        "PRON": "function",
        "PRT": "function",
        "ADJ": "content",
        "ADV": "content",
        "NOUN": "content",
        "NUM": "content",
        "VERB": "content",
        "X": "unknown",
    }
    function_content = df.apply(
        lambda x: function_content_dict.get(x["part_of_speech"]), axis=1
    )
    df = df.assign(function_content=function_content)

    return df


def add_phoneme(df, dirname):
    # get phoneme dict
    cmu_dict_filename = f"{dirname}cmudict-0.7b"
    pdict = {}
    with open(cmu_dict_filename, "r", encoding="ISO-8859-1") as f:
        for line in f.readlines():
            if not line.startswith(";;;"):
                parts = line.rstrip().split()
                word = parts[0].lower()
                phones = [phone.rstrip("012") for phone in parts[1:]]
                pdict[word] = phones

    words2phonemes = df.apply(lambda x: pdict.get(x["word"].lower()), axis=1)

    # add to df
    df = df.assign(pho=words2phonemes)
    df = df[~df.pho.isnull()]
    df = df.explode("pho", ignore_index=False)
    df["pho_idx"] = df.groupby(["word", "adjusted_onset"]).cumcount() + 1

    return df


def add_phoneme_cat(df, dirname):
    # original categorization, including specific vowel catergorization
    # phoneset = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY', 'F' , 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L',  'M', 'N' , 'NG', 'OW', 'OY', 'P',  'R', 'S',  'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH']
    # place_of_articulation   = ['low-central', 'low-front', 'mid-central', 'mid-back', 'high-back', 'high-front', 'bilabial', 'post-alveolar', 'alveolar', 'inter-dental', 'mid-front', 'mid-central', 'mid-front','alveolar','velar', 'glotal', 'high-front', 'high-front', 'post-alveolar', 'velar', 'alveolar', 'bilabial', 'alveolar', 'velar', 'high-back', 'high-front', 'bilabial', 'alveolar', 'alveolar', 'post-alveolar', 'alveolar', 'inter-dental', 'high-back', 'high-back', 'labio-dental', 'bilabial', 'palatal', 'alveolar', 'post-alveolar']
    # manner_of_articulation  = ['lax', 'lax', 'lax', 'lax', 'lax', 'tense', 'stop', 'affricate', 'stop', 'fricative', 'lax', 'tense', 'tense', 'flap', 'stop','fricative', 'lax', 'tense', 'affricate', 'stop', 'lateral-liquid', 'nasal', 'nasal', 'nasal', 'lax', 'lax', 'stop', 'retroflex-liquid', 'fricative', 'fricative', 'stop', 'fricative', 'lax', 'tense', 'fricative', 'glide', 'glide', 'fricative', 'fricative']

    # create categorizations
    phoneset_categorizations = pd.read_csv(f"{dirname}phoneset.csv")
    phoneset = phoneset_categorizations.Phoneme.values
    place_of_articulation = phoneset_categorizations.iloc[:, 1].values
    manner_of_articulation = phoneset_categorizations.iloc[:, 2].values
    voiced_or_voiceless = phoneset_categorizations.iloc[:, 3].values

    place_of_articulation_dict = dict(zip(phoneset, place_of_articulation))
    manner_of_articulation_dict = dict(zip(phoneset, manner_of_articulation))
    voiced_or_voiceless_dict = dict(zip(phoneset, voiced_or_voiceless))

    phocat = df.apply(lambda x: place_of_articulation_dict.get(x["pho"]), axis=1)
    df = df.assign(poa=phocat)
    phocat = df.apply(lambda x: manner_of_articulation_dict.get(x["pho"]), axis=1)
    df = df.assign(moa=phocat)
    phocat = df.apply(lambda x: voiced_or_voiceless_dict.get(x["pho"]), axis=1)
    df = df.assign(voi=phocat)

    return df


def get_emb_speech(df):
    features = [PHO_LIST, POA_LIST, MOA_LIST, VOI_LIST]
    cols = ["pho", "poa", "moa", "voi"]
    embs = []
    for feature, col in zip(features, cols):
        ohe = OneHotEncoder(categories=[feature], drop="if_binary")
        emb = ohe.fit_transform(df[col].values.reshape(-1, 1)).toarray()
        embs.append(emb)
    embs = np.hstack(embs)

    df["embeddings"] = embs.tolist()  # should be 60

    def concat_speech_emb(subdf):
        embs = np.array(subdf.embeddings.tolist()).flatten()
        embs = np.tile(embs, [len(subdf), 1])
        subdf = subdf.assign(embeddings=embs.tolist())
        return subdf

    df = df.groupby(["word", "onset", "offset"]).apply(concat_speech_emb)

    return df


def get_emb_lang(df):
    features = [
        POS_LIST_FROM_SPACY,
        DEP_LIST_FROM_SPACY,
        ENGLISH_PREFIXES + [""],
        ENGLISH_SUFFIXES + [""],
        [False, True],
    ]
    cols = ["pos", "dep", "prefix", "suffix", "is_stop"]
    embs = []
    for feature, col in zip(features, cols):
        ohe = OneHotEncoder(categories=[feature], drop="if_binary")
        emb = ohe.fit_transform(df[col].values.reshape(-1, 1)).toarray()
        if "" in feature:
            print(f"skipping empty for {col}")
            emb = emb[:, :-1]
        embs.append(emb)
    embs = np.hstack(embs)
    assert embs.shape[0] == len(df)
    df["embeddings"] = embs.tolist()  # should be 137

    return df


def clean(df):  # Some random issue where word_without_punc is uppercased
    df = df.assign(
        word_without_punctuation=df["word"].apply(
            lambda x: x.translate(
                str.maketrans("", "", '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~')
            )
        )
    )
    return df


def generate_symbolic_embeddings(args, df):
    df = clean(df)  # fix some bug from pickling

    ###### speech symbolic #####
    if args.embedding_type == "symbolic-speech":
        # phoneme (http://www.speech.cs.cmu.edu/cgi-bin/cmudict)(37 out of 39)
        df = add_phoneme(df, "results/tfs/")

        # place of articulation (9 out of 9)
        # manner of articulation (9 out of 9)
        # voiced or voiceless (3 out of 3)
        df = add_phoneme_cat(df, "results/tfs/")

        # get symbolic embedding
        df = get_emb_speech(df)

    ###### language symbolic #####
    elif args.embedding_type == "symbolic-lang":
        # part of speech tag (https://www.nltk.org/book/ch05.html) (11 out of 12)
        # function content (3 out of 3)
        df = add_pos_nltk(df)

        # prefix (30) & suffix (44)
        df = get_prefix_suffix(df)  # add prefix/suffix (manual)

        # shape ? how many
        # is_stop (1, binary)
        # dependency (https://spacy.io/models/en) (45)
        # part of speech tag (https://universaldependencies.org/u/pos/) (out of 17)
        df = get_semantic_info(df)

        # get symbolic embedding
        df = get_emb_lang(df)

    return df
