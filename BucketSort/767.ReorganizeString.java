/*
	Remember to append the maxChar first in each every StringBuilder
	Because it could be the case that two maxChar get appended in the same bucket
	when other char are all used up
	NOTICE:
		Should always sepparate maxChar as scattered as possible!
*/
class Solution {
    public String reorganizeString(String S) {
        int[] letters = new int[26];
        int max = 0;
        char maxChar = 'a';
        for (char c: S.toCharArray()) {
            letters[c-'a']++;
            if (max < letters[c-'a']) {
                maxChar = c;
                max = letters[c-'a'];
            }
        }
        if (max == 1) return S;
        if (S.length() <= 2*max - 2) return "";
        StringBuilder[] sbs = new StringBuilder[max];
        for (int i = 0; i < sbs.length; i++) {
            sbs[i] = new StringBuilder();
            sbs[i].append(maxChar);
        }
        int idx = 0;
        for (char c = 'a'; c <= 'z'; c++) {
            while (c != maxChar && letters[c-'a'] > 0) {                
                sbs[idx++].append(c);
                letters[c-'a']--;
                idx %= max;
            }
        }
        for (int i = 1; i < max; i++)
            sbs[0].append(sbs[i]);
        return sbs[0].toString();
    }
}