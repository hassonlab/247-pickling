import numpy as np
import pandas as pd
import nltk
import spacy
import re
import string

# from nltk.corpus import words
# from nltk.corpus import wordnet as wn
# from nltk.stem import PorterStemmer


def get_prefix_suffix(df):
    english_prefixes = [
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

    noun_suffixes = [
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

    verb_suffixes = [
        "ate",  # e.g. complicate, dominate, irritate
        "en",  # e.g. harden, soften, shorten
        "ify",  # e.g. beautify, clarify, identify
        "ise",  # e.g. economise, realise (-ise is most common in British English) 1
        "ize",  # e.g. industrialize (-ize is most common in American English) 1
    ]

    adj_suffixes = [
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

    adv_suffixes = [
        "ly",  # e.g. calmly, easily, quickly
        "ward",  # e.g. downward, homeward, upward 1
        "wards",  # e.g. downwards, homeward(s), upwards 1
        "wise",  # e.g. anti-clockwise, clockwise, edgewise
    ]

    prefix_list = ["ADV", "ADJ", "VERB", "NOUN"]

    def get_prefix(word):
        for prefix in sorted(english_prefixes, key=len, reverse=True):
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
    df = get_suffix(df, "NOUN", noun_suffixes)
    df = get_suffix(df, "VERB", verb_suffixes)
    df = get_suffix(df, "ADJ", adj_suffixes)
    df = get_suffix(df, "ADV", adv_suffixes)

    df.prefix = df.prefix.fillna("").astype(str)
    df.suffix = df.suffix.fillna("").astype(str)

    return df


def get_spacy_stuff(df):
    sp = spacy.load("en_core_web_sm")

    def get_shape(word):  # get shape
        doc = sp(word)
        return doc[0].shape_

    def get_stop(word):  # get stop
        doc = sp(word)
        return doc[0].is_stop

    df["shape"] = df.word_without_punctuation.apply(get_shape)
    df["is_stop"] = df.word_without_punctuation.apply(get_stop)

    return df


def add_speech(whisper_df):
    # Get Part of Speech
    words_orig, part_of_speech = zip(
        *nltk.pos_tag(whisper_df.word, tagset="universal")
    )
    whisper_df = whisper_df.assign(part_of_speech=part_of_speech)

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
    # function_content_dict = {
    #     "ADP": "function",
    #     "CONJ": "function",
    #     "DET": "function",
    #     "PRON": "function",
    #     "PRT": "function",
    #     "ADJ": "content",
    #     "ADV": "content",
    #     "NOUN": "content",
    #     "NUM": "content",
    #     "VERB": "content",
    #     "X": "unknown",
    # }
    # function_content = whisper_df.apply(
    #     lambda x: function_content_dict.get(x["part_of_speech"]), axis=1
    # )
    # whisper_df = whisper_df.assign(function_content=function_content)

    return whisper_df


def zeroshot_datum(df):
    dfz = (
        df[["word_without_punctuation", "adjusted_onset"]]
        .groupby("word_without_punctuation")
        .apply(lambda x: x.sample(1, random_state=42))
    )
    dfz.reset_index(level=1, inplace=True)
    dfz.sort_values("adjusted_onset", inplace=True)
    df = df.loc[dfz.level_1.values]
    print(f"Zeroshot created datum with {len(df)} words")

    return df


def rand_emb(df):
    # emb_max = df.embeddings.apply(max).max()
    # emb_min = df.embeddings.apply(min).min()

    rand_emb = np.random.random((len(df), 50))
    # rand_emb = rand_emb * (emb_max - emb_min) + emb_min
    df2 = df.copy()  # setting copy to avoid warning
    df2["embeddings"] = list(rand_emb)
    df = df2  # reassign back to datum
    print(f"Generated random embeddings for {len(df)} words")

    return df


def arb_emb(df):
    df2 = zeroshot_datum(df)
    df2 = df2.loc[:, ("word_without_punctuation", "datum_word")]
    df2.reset_index(drop=True, inplace=True)
    df2 = rand_emb(df2)
    df = df.drop("embeddings", axis=1, errors="ignore")

    df = df.merge(df2, how="left", on="word_without_punctuation")
    print(f"Arbitrary embeddings created for {len(df)} words")

    return df


def get_emb(df):
    # df["next_emb"] = df.embeddings.shift(-1)  # next word
    # df["prev_emb"] = df.embeddings.shift(1)  # next word
    # df["prev_tag"] = df.part_of_speech.shift(1)  # previous tag
    # df["prev_tag2"] = df.part_of_speech.shift(2)  # previous two tag
    # df.dropna(
    # subset=["embeddings", "next_emb", "prev_emb", "prev_tag", "prev_tag2"],
    # inplace=True,
    # )

    current_emb = np.stack(df.embeddings)
    # next_emb = np.stack(df.next_emb)
    # prev_emb = np.stack(df.prev_emb)
    one_hot_cur_tag = np.array(pd.get_dummies(df.part_of_speech))
    # one_hot_prev_tag = np.array(pd.get_dummies(df.prev_tag))
    # one_hot_prev_tag2 = np.array(pd.get_dummies(df.prev_tag2))
    one_hot_prefix = np.array(pd.get_dummies(df.prefix))[:, 1:]
    one_hot_suffix = np.array(pd.get_dummies(df.suffix))[:, 1:]
    # one_hot_prefix = np.array(pd.get_dummies(df.prefix))  # used for 11
    # one_hot_suffix = np.array(pd.get_dummies(df.suffix))  # used for 11
    one_hot_shape = np.array(pd.get_dummies(df["shape"]))
    one_hot_stop = np.array(pd.get_dummies(df.is_stop))[:, 1:]
    # one_hot_stop = np.array(pd.get_dummies(df.is_stop))  # used for 11

    # def get_cat(col_name):
    #     return np.reshape(
    #         np.array(df[col_name].astype("category").cat.codes), (-1, 1)
    #     )

    # cat_cur_tag = get_cat("part_of_speech")
    # cat_stop = get_cat("is_stop")
    # cat_shape = get_cat("shape")
    # cat_prefix = get_cat("prefix")
    # cat_suffix = get_cat("suffix")

    final_emb = np.concatenate(
        (
            # prev_emb,
            # current_emb,
            # next_emb,
            # one_hot_prev_tag2,
            # one_hot_prev_tag,
            # one_hot_cur_tag,
            # one_hot_stop,
            # one_hot_shape,
            # one_hot_prefix,
            one_hot_suffix,
            # cat_cur_tag,
            # cat_stop,
            # cat_shape,
            # cat_prefix,
            # cat_suffix,
        ),
        axis=1,
    )
    print(
        f"Concatenating:",
        f"\ncurrent emb ({current_emb.shape[1]})",  # 50
        # f"\nprev emb ({prev_emb.shape[1]})",  # 50
        # f"\nnext emb ({next_emb.shape[1]})",  # 50
        # f"\nprevious pos tag ({one_hot_prev_tag.shape[1]})",  # 11
        # f"\nprevious 2 pos tag ({one_hot_prev_tag2.shape[1]})",  # 11
        f"\ncurrent pos tag ({one_hot_cur_tag.shape[1]})",  # 11
        f"\nis_stop ({one_hot_stop.shape[1]})",  # 1
        f"\nshape ({one_hot_shape.shape[1]})",  # 16
        f"\nprefix ({one_hot_prefix.shape[1]})",  # 18
        f"\nsuffix ({one_hot_suffix.shape[1]})",  # 28
        f"\nFinal Emb Size ({final_emb.shape[1]})",
    )
    df["embeddings"] = list(final_emb)

    return df


def clean(df):  # Some random issue where word_without_punc is uppercased
    df.loc[:, "word_without_punctuation"] = df["word"].apply(
        lambda x: x.translate(
            str.maketrans("", "", '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~')
        )
    )
    return df


def generate_symbolic_embeddings(df):
    df = clean(df)  # fix some bug from pickling
    df = arb_emb(df)  # random emb
    df = add_speech(df)  # add tag (nltk)
    df = get_prefix_suffix(df)  # add prefix/suffix (manual)
    df = get_spacy_stuff(df)
    breakpoint()
    df = get_emb(df)

    return df
