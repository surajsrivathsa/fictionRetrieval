package org.ovgu.de.fiction.feature.extraction;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.ovgu.de.fiction.model.Concept;
import org.ovgu.de.fiction.model.Word;
import org.ovgu.de.fiction.utils.FRConstants;
import org.ovgu.de.fiction.utils.FRGeneralUtils;
import org.ovgu.de.fiction.utils.FRFileOperationUtils;
import org.ovgu.de.fiction.utils.StanfordPipeline;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.util.CoreMap;

/**
 * @author Suhita, Sayantan
 * @version - Changes for sentiment features
 */
public class WordAttributeGenerator {

	private static final String NNP = "NNP";

	/**
	 * @param path
	 *            = path of whole book, not chunk
	 * @return = List of "Word" objects, each object has original token, POS tag, lemma, NER as
	 *         elements
	 * @author Suhita, Modified by Sayantan for # of characters
	 */
	
	/*
	 * @suraj: 
	 */
	
	 public Concept generateWordAttributes(Path path) {

		FeatureExtractorUtility feu = new FeatureExtractorUtility();
		Concept cncpt = new Concept();
		Annotation document = new Annotation(FRFileOperationUtils.readFile(path.toString()));

		StanfordPipeline.getPipeline(null).annotate(document);
		
		// @suraj: split the document into an array of sentences using stanford nlp library
		
		List<CoreMap> sentences = document.get(CoreAnnotations.SentencesAnnotation.class);
		
		List<Word> tokenList = new ArrayList<>();
		Map<String, Integer> charMap = new HashMap<>(); // a new object per new
														 // book // Book
		StringBuffer charName = new StringBuffer();
		int numOfSyllables = 0;
		int numOfSentences =0;

		for (CoreMap sentence : sentences) { // this loop will iterate each of the sentences
			
			// @suraj : Word object has actual word, lemma, pos, ner, #sylabble
			//S_tag : <s>. Annotate the begining of sentence with <s>
			tokenList.add(new Word(FRConstants.S_TAG, FRConstants.S_TAG, null, null, 0));
			numOfSentences++;   //increment sentence count
			
			//@ suraj: For each token in sentence
			for (CoreLabel cl : sentence.get(CoreAnnotations.TokensAnnotation.class)) {			
				// @suraj: for each sentence
				// @suraj: extract actual token, partsofspeech, named entity(real-world object, such as persons, locations, organizations, products) and lemmatized representation
				
				String original = cl.get(CoreAnnotations.OriginalTextAnnotation.class);
				String pos = cl.get(CoreAnnotations.PartOfSpeechAnnotation.class);
				String ner = cl.get(CoreAnnotations.NamedEntityTagAnnotation.class);
				String lemma = cl.get(CoreAnnotations.LemmaAnnotation.class).toLowerCase();
				
				/* 
				 * @sayantan: 
				 * logic 2: check if ner is "P", then further check next 2 element in sentence , ex.
				 * Tom Cruise, Mr. Tom Cruise if yes, then concatenate all two or three tokens i.e.
				 * "Mr" +"Tom" + "Cruise" into a single token the single concatenated token is added
				 * to a Map , where key is number of times "Mr. Tom Cruise" appears
				 */
				
				if (ner.equals(FRConstants.NER_CHARACTER) && !original.matches(FRConstants.CHARACTER_STOPWORD_REGEX)) {
					if (charName.length() == 0)
						charName.append(original.toLowerCase());
					else
						charName.append(FRConstants.SPACE).append(original.toLowerCase());
				} 
				else if (!ner.equals(FRConstants.NER_CHARACTER) && charName.length() != 0) {

					/* @suraj: calculate for syllables
					 *  Mr. Tom Cruise ==> Mr Tom Cruise ==> MrTomCruise
					 *  We are keeping track of plot characters in the book through NNP tag and finding count of each character appearance
					 */
					
					numOfSyllables = FRGeneralUtils.countSyllables(charName.toString().toLowerCase());
					addToTokenList(tokenList, charName.toString(), NNP, FRConstants.NER_CHARACTER, charName.toString(), numOfSyllables);
					
					// @suraj: Mr. Tom Cruise ==> Mr Tom Cruise ==> MrTomCruise
					
					String charNameStr = charName.toString().replaceAll(FRConstants.REGEX_ALL_PUNCTUATION, " ")
							.replaceAll(FRConstants.REGEX_TRAILING_SPACE, "");
					
					// @suraj: Number of times mrtomcruise appears in hashmap dictionary
					int count = charMap.containsKey(charNameStr) ? charMap.get(charNameStr) : 0;
					charMap.put(charNameStr, count + 1);

					// add the next word after character
					addToTokenList(tokenList, original, pos, ner, lemma, FRGeneralUtils.countSyllables(original.toLowerCase()));
					// rest the string buffer
					charName = new StringBuffer();

				} 
				else {
					addToTokenList(tokenList, original, pos, ner, lemma, FRGeneralUtils.countSyllables(original.toLowerCase()));
				}

			}
		}
		cncpt.setWords(tokenList);
		cncpt.setCharacterMap(feu.getUniqueCharacterMap(charMap));
		cncpt.setNumOfSentencesPerBook(numOfSentences);
		StanfordPipeline.resetPipeline();
		return cncpt;
	}

	/*
	 * @suraj: Input: word along with its lemma, pos, #sylables
	 * Check whether each word is not special ie: word contains single quotes ' like eight 'o clock, or
	 * words contain a period after them like mr. and mrs. Handle these cases by removing period and quotes and add to token
	 * list, where each word token will have original word, lemma, pos , name entity, #sylables
	 */
	
	public void addToTokenList(List<Word> tokenList, String original, String pos, String ner, String lemma, int numOfSyllbles) {
		if (lemma.matches("^'.*[a-zA-Z]$")) { // 's o'clock 'em, ain't => ain+t
			StringBuffer sbf = new StringBuffer();
			Arrays.stream(lemma.split("'")).forEach(l -> sbf.append(l));
			tokenList.add(new Word(original, sbf.toString(), pos, ner, numOfSyllbles));
		} // mr. mrs. mr + ""
		else if (lemma.matches("[a-zA-Z0-9].*[.].*") && ner.matches("(O|MISC)")) {
			tokenList.add(new Word(original, lemma.split("\\.")[0], pos, ner, numOfSyllbles));
		} else {
			tokenList.add(new Word(original, lemma, pos, ner, numOfSyllbles));
		}
	}

}
