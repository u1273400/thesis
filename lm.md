<div id="table-of-contents">
<h2>Table of Contents</h2>
<div id="text-table-of-contents">
<ul>
<li><a href="#sec-1">1. <span class="todo TODO">TODO</span> Introduction</a></li>
<li><a href="#sec-2">2. <span class="todo TODO">TODO</span> Literature</a>
<ul>
<li><a href="#sec-2-1">2.1. <span class="todo TODO">TODO</span> Word Sequence LSTM</a></li>
<li><a href="#sec-2-2">2.2. <span class="todo TODO">TODO</span> n-gram Models</a></li>
<li><a href="#sec-2-3">2.3. <span class="todo TODO">TODO</span> Random papers x2</a></li>
</ul>
</li>
<li><a href="#sec-3">3. <span class="todo TODO">TODO</span> Method</a>
<ul>
<li><a href="#sec-3-1">3.1. <span class="todo TODO">TODO</span> Unreasonable Effectiveness</a></li>
<li><a href="#sec-3-2">3.2. <span class="todo TODO">TODO</span> The Dataset</a></li>
<li><a href="#sec-3-3">3.3. <span class="todo TODO">TODO</span> LSTM Training</a></li>
</ul>
</li>
<li><a href="#sec-4">4. Results</a></li>
<li><a href="#sec-5">5. References</a></li>
</ul>
</div>
</div>

\begin{abstract}

This is the abstract.

\end{abstract}

# TODO Introduction<a id="sec-1" name="sec-1"></a>

# TODO Literature<a id="sec-2" name="sec-2"></a>

## TODO Word Sequence LSTM<a id="sec-2-1" name="sec-2-1"></a>

## TODO n-gram Models<a id="sec-2-2" name="sec-2-2"></a>

## TODO Random papers x2<a id="sec-2-3" name="sec-2-3"></a>

# TODO Method<a id="sec-3" name="sec-3"></a>

## TODO Unreasonable Effectiveness<a id="sec-3-1" name="sec-3-1"></a>

## TODO The Dataset<a id="sec-3-2" name="sec-3-2"></a>

## TODO LSTM Training<a id="sec-3-3" name="sec-3-3"></a>

# Results<a id="sec-4" name="sec-4"></a>

For this research work, a corpus of the Okrika Language having a word size of 113k words of the New Testament Gospels in Okrika language.  The vocabulary had a word size of appriximately 5000 words.  The corpus generated from the original corpus using the LSTM-RNN character sequence network produced a word size of 118k words. However the vocabulary of the output LSTM corpus almost doubled to 9000 words. 

The result of the training of the Long-short-term-memory (LSTM)-Cell Recurrent Neural Network on low-resourced Okrika Language gave impressive and intelligble results and showed competitive results when measured with Standard n-gram language models.  The results showed that it is indeed possible to use an LSTM on a low resource character sequence corpus to produce an Okrika language Generator.

The evaluation of the LSTM language model of the Okrika language done using a perplexity measurement metric.  The Perplexity metric applies the language model to a test dataset and measures how probable the test dataset is.  Perplexity is a relative measure given by the formula:$
$\begin{aligned}PP(W)&=P(w<sub>1</sub>,w<sub>2&hellip;</sub> w<sub>N</sub>)<sup>frac</sup>{1}{N}
\\\\&=sqrt{N}{&prod;<sub>i=1</sub><sup>N\frac</sup>{1}{P(w<sub>i|w</sub><sub>i-1</sub>}}
\end{aligned}$$

Therefore language models with higher perplexity are expected to yield better approximation of the language data when applied to the language generally.

There is no way however to directly measure perplexity on a character sequence model because perplexity is usually used to evaluate word-based models.  However, this limitation was overcome by performing n-gram analysis on the corpus entirely generated from the LSTM network. The generated n-gram model from the generated corpus is then applied to test data and the perplexity is measured.

Table 1 below shows the Results of the Perplexity model of the LSTM Okrika Language model and an equivalent Tri-gram Language model with interpolation and Keysner smoothing.

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="left" />

<col  class="right" />
</colgroup>
<thead>
<tr>
<th scope="col" class="left">Language Model</th>
<th scope="col" class="right">Perplexity</th>
</tr>
</thead>

<tbody>
<tr>
<td class="left">LSTM RNN</td>
<td class="right">2.6</td>
</tr>


<tr>
<td class="left">3-gram with Keysner Soothing and interpolation</td>
<td class="right">3.3</td>
</tr>
</tbody>
</table>

Although the n-gram had better perplexity measurement the LSTM showed promising results and because it is based on a character-model, which is fine-grained when compared to a word model, it is likely to generalise data better when used in practice and therefore less biased than a word-based model.  This can be observed from the fact that the output corpus produced a larger vocabulary size.

# References<a id="sec-5" name="sec-5"></a>

references:bib.bib