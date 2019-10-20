/**
Similar idea as permutation II. 
Since every time we are adding different char at the front and end of the sb,
we don't need to consider when there is duplicated char.

*/

class Solution {
    public List<String> generatePalindromes(String s) {
        List<String> ans = new ArrayList<>();
        int[] letters = new int[256];
        for (char c: s.toCharArray()) {
            letters[c]++;
        }
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < letters.length; i++)
            if (letters[i] % 2 == 1) sb.append((char) i);        
        if(sb.length() > 1) return ans;
        permutation(ans, letters, sb, s.length());
        return ans;
    }
    
    private void permutation(List<String> ans, int[] letters, StringBuilder sb, int len) {
        if (sb.length() == len) ans.add(new String(sb));
        for (int i = 0; i < letters.length; i++) {
            if (letters[i] > 1) {
                char c = (char) i;
                sb.append(c);
                sb.insert(0, c);                
                letters[i] -= 2;
                permutation(ans, letters, sb, len);
                letters[i] += 2;
                sb.deleteCharAt(sb.length() - 1);
                sb.deleteCharAt(0);
            }
        }
    }
}