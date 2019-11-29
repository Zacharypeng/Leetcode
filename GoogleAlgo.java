/*
	169. Majority Element
	Given an array of size n, find the majority element. The majority element is the element that appears more than ⌊ n/2 ⌋ times.

	You may assume that the array is non-empty and the majority element always exist in the array.
*/
class Solution {
    public int majorityElement(int[] nums) {
        int cnt=0, target=nums[0];
        for (int i = 0; i < nums.length; i++) {
            if (target==nums[i]) cnt++;
            else cnt--;
            if (cnt <= 0) {
                target = nums[i];
                cnt=1;
            }
        }
        return target;
    }
}

/*
	229. Majority Element II
	len // 3
	Remeber to double check the final result is bigger than len // 3 !!
*/
public List<Integer> majorityElement(int[] nums) {
        List<Integer> res = new ArrayList<>();
        if(nums.length == 0)
            return res;
            
        int num1 = nums[0]; int num2 = nums[0]; int count1 = 1; int count2 = 0 ;
        
        for (int val : nums) {
            if(val == num1)
                count1++;
            else if (val == num2)
                count2++;
            else if (count1 == 0) {
                num1 = val;
                count1++;
                }
            else if (count2 == 0) {
                num2 = val;
                count2++;
            }
            else {
                count1--;
                count2--;
            }
        }
        count1 = 0;
        count2 = 0;
        for(int val : nums) {
            if(val == num1)
                count1++;
            else if(val == num2)
                count2++;
        }
        if(count1 > nums.length/3)
            res.add(num1);
        if(count2 > nums.length/3)
            res.add(num2);
        return res;
    }	

/*
	465. Optimal Account Balancing
	[0, 1, 10], 0 -> 1 10 bucks
	return minimum number of transactions to settle down

	Notice: 
		into the dfs, debt.get(i) + debt.get(start) is to settle down the start
*/    
class Solution {
    public int minTransfers(int[][] transactions) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int[] t: transactions) {
            map.put(t[0], map.getOrDefault(t[0], 0) - t[2]);
            map.put(t[1], map.getOrDefault(t[1], 0) + t[2]);
        }
        int res = settle(0, new ArrayList<>(map.values()));
        return res;
    }
    
    private int settle(int start, ArrayList<Integer> debt) {
        while (start < debt.size() && debt.get(start) == 0)
            start++;
        if (start == debt.size()) return 0;
        int res = Integer.MAX_VALUE;
        for (int i = start + 1; i < debt.size(); i++) {
            if (debt.get(i) * debt.get(start) < 0) {
                debt.set(i, debt.get(i) + debt.get(start));
                res = Math.min(res, 1 + settle(start + 1, debt));
                debt.set(i, debt.get(i) - debt.get(start));
            }
        }
        return res;
    }
}	

/*
	729. My Calendar I
	MyCalendar cal = new MyCalendar(); MyCalendar.book(start, end)
	return if there is overlapping event
*/
class MyCalendar {
    TreeMap<Integer, Integer> map;
    public MyCalendar() {
        map = new TreeMap<Integer, Integer>();
    }
    
    public boolean book(int start, int end) {
        Integer floorKey = map.floorKey(start);
        if (floorKey != null && map.get(floorKey) > start) return false;
        Integer ceilingKey = map.ceilingKey(start);
        if (ceilingKey != null && ceilingKey < end) return false;
        map.put(start, end);
        return true;
    }
}

/*
	732. My Calendar III
	return the maximum overlapping event
	same idea as meeting rooms II
*/

class MyCalendarThree {
    TreeMap<Integer, Integer> map;
    public MyCalendarThree() {
        map = new TreeMap<>();
    }
    
    public int book(int start, int end) {
        map.put(start, map.getOrDefault(start, 0) + 1);
        map.put(end, map.getOrDefault(end, 0) - 1);
        int res = 0, cur = 0;
        for (int cnt: map.values()) {
            res = Math.max(res, cur += cnt);
        }
        return res;
    }
}	


/*
	91. Decode Ways
	'A' -> 1
	'B' -> 2
	...
	'Z' -> 26
	Input: "226"
	Output: 3
	Explanation: It could be decoded as "BZ" (2 26), "VF" (22 6), or "BBF" (2 2 6).

	dp[i] means how many ways to decode for the substing(0, i)
	Notice:
		dp[0] should be 1 !!
*/
public int numDecodings(String s) {
        int[] dp = new int[s.length() + 1];
        dp[0] = 1;
        dp[1] = s.charAt(0) != '0' ? 1 : 0;
        for (int i = 2; i <= s.length(); i++) {
            int first = Integer.parseInt(s.substring(i-1, i));
            int second = Integer.parseInt(s.substring(i-2, i));
            if (first >= 1 && first <=9)
                dp[i] += dp[i-1];
            if (second >=10 && second <= 26)
                dp[i] += dp[i-2];
        }
        return dp[s.length()];
    }

/*
	337. House Robber III
	Rober tree
	rob the root and don't rob it's left and right child OR
	rob the left and right child and don't rob the root

	Notice: 
		if don't rob the root, res[0] choose the max from left and right
*/    
class Solution {
    // res[0] don't rob this node
    // res[1] rod this node and don't rob its left and right node
    public int rob(TreeNode root) {
        if(root == null) return 0;
        int[] res = helper(root);
        return Math.max(res[0], res[1]);
    }
    
    private int[] helper(TreeNode root) {
        if (root == null) return new int[2];
        int[] left = helper(root.left);
        int[] right = helper(root.right);
        
        int[] res = new int[2];
        res[0] = Math.max(left[0], left[1]) + Math.max(right[0], right[1]);
        res[1] = root.val + left[0] + right[0];
        return res;
    }
}	


/*
	1032. Stream of Characters
	
	Notice:
		if encounter a true word, still need to add it to the queue !!
		Because we don't know whether this word is a leaf
		Pay attention to the edge case when node == root! DO NOT remove the root from the queue!
*/
class StreamChecker {
    class Trie {
        boolean isWord;
        Trie[] children = new Trie[26];
    }
    Trie root = new Trie();
    Queue<Trie> q = new LinkedList<>();
    public StreamChecker(String[] words) {
        for (String word: words) {
            Trie node = root;
            for (char c: word.toCharArray()) {
                if (node.children[c-'a'] == null) {
                    node.children[c-'a'] = new Trie();
                }
                node = node.children[c-'a'];
            }
            node.isWord = true;
        }
        q.add(root);
    }
    
    public boolean query(char letter) {        
        boolean res = false;
        int size = q.size();
        while (size-- > 0) {
            Trie node = q.poll();
            if (node.children[letter-'a'] != null) {
                Trie temp = node.children[letter-'a'];                
                if (temp.isWord) {
                    res = true;
                }
                q.add(node.children[letter-'a']);
            }
            if (node == root) 
                q.add(node);
        }
        return res;
    }
}


/*
	295. Find Median from Data Stream

	everything is "reversed"
	Notice:
		size of Min will always be equal or greater than Max
		num less than the median goes to Max
		change to Double when it's even number !!!
*/		
class MedianFinder {

    /** initialize your data structure here. */
    PriorityQueue<Integer> minQue;
    PriorityQueue<Integer> maxQue;
    public MedianFinder() {
        minQue = new PriorityQueue<>();
        maxQue = new PriorityQueue<>( (a, b) -> (b.compareTo(a)));
    }
    
    public void addNum(int num) {
        if (num < findMedian()) maxQue.add(num);
        else minQue.add(num);
        if (maxQue.size() > minQue.size()) 
            minQue.add(maxQue.poll());
        if (minQue.size() - maxQue.size() > 1)
            maxQue.add(minQue.poll());
    }
    
    public double findMedian() {
        if (minQue.isEmpty() && maxQue.isEmpty()) return 0.0;
        if (minQue.size() == maxQue.size())
            return (double)((double)minQue.peek() + (double)maxQue.peek()) / 2.0;
        else
            return (double)minQue.peek();
    }
}		


/*
	792. Number of Matching Subsequences
	Input: 
	S = "abcde"
	words = ["a", "bb", "acd", "ace"]
	Output: 3

	Notes:
		iterate through the string and greedly remove the first char of every candidate string

	Notice:
		by adding the whole string in the value avoid the edge case when candidates only have one char
		the List can be replaced by a Deque, using addLast and removeFirst
*/
class Solution {
    public int numMatchingSubseq(String S, String[] words) {
        Map<Character, List<String>> map = new HashMap<>();
        for (char c: S.toCharArray()) {
            map.putIfAbsent(c, new ArrayList<String>());
        }
        for (String word: words) {
            if (map.containsKey(word.charAt(0)))
                map.get(word.charAt(0)).add(word);
        }
        int cnt = 0;
        for (int i = 0; i < S.length(); i++) {
            char c = S.charAt(i);
            if (map.containsKey(c)) {
                List<String> temp = map.get(c);
                int size = temp.size();
                for (int j = 0; j < size; j++) {
                    String t = temp.remove(0);
                    if (t.length() == 1) cnt ++;
                    else {
                        if (map.containsKey(t.charAt(1))) {
                            map.get(t.charAt(1)).add(t.substring(1));
                        }
                    }
                }
            }
        }
        return cnt;
    }
}

/*
	752. Open the Lock
	Start from "0000", change to target
	if meet deadend, will stuck. '0' <- '9', '9' -> '0'

	This problem can be solved by a basic BFS
	This solution is for BFS form both ends
	the new element get added to temp, begine change to end and temp change to end
	Notice:
		the way to change one char(use int instead)
*/		
class Solution {
    public int openLock(String[] deadends, String target) {
        Set<String> dead = new HashSet<>();        
        for (String s: deadends)
            dead.add(s);
        Queue<String> begine = new LinkedList<>();
        Queue<String> end = new LinkedList<>();
        begine.add("0000");
        end.add(target);        
        int level = 0;
        while (!begine.isEmpty() && !end.isEmpty()) {
            Queue<String> temp = new LinkedList<>();
            int size = begine.size();
            for (int k = 0; k < size; k++) {
                String s = begine.poll(); 
                if (dead.contains(s)) continue;
                dead.add(s);
                if (end.contains(s)) return level;
                for (int i = 0; i < 4; i++) {
                    char c = s.charAt(i);
                    String s1 = s.substring(0, i) + (c == '9' ? 0 : c - '0' + 1) + s.substring(i+1);
                    String s2 = s.substring(0, i) + (c == '0' ? 9 : c - '0' - 1) + s.substring(i+1);
                    if (!dead.contains(s1)) {
                        temp.add(s1);
                        // dead.add(s1);
                    }
                    if (!dead.contains(s2)) {
                        temp.add(s2);
                        // dead.add(s2);
                    }
                }                
            }            
            begine = end;
            end = temp;
            level++;
        }
        return -1;
    }
}	


/*
	809. Expressive Words
	Input: 
	S = "heeellooo"
	words = ["hello", "hi", "helo"]
	Output: 1
	"helo" can't change to "hello", because in the original string, "ll" is not size 3 or more

	Notice:
		in the check, when going out side of the loop
		REMEMBER to check whether i and j is at the end of each string!!!
*/
class Solution {
    public int expressiveWords(String S, String[] words) {
        int cnt = 0;
        for (String word: words) {
            if (check(S, word)) cnt++;
        }
        return cnt;
    }

    private boolean check(String a, String b) {
    	int i = 0, i2 = 0, j = 0, j2 = 0;
    	while (i < a.length() && j < b.length()) {
    		if (a.charAt(i) != b.charAt(j)) return false;
    		while (i2 < a.length() && a.charAt(i) == a.charAt(i2)) i2++;
    		while (j2 < b.length() && b.charAt(j) == b.charAt(j2)) j2++;
    		if (i2 - i != j2 - j && i2 - i < Math.max(3, j2 - j)) return false;
            i = i2;
            j = j2; 
    	}
    	return i == a.length() && j == b.length();
    }
}	


/*
	843. Guess the Word

	Use random to randomly select a word in candidate array
	update the array to only contains string that has the exact number of match 
*/
class Solution {
    public void findSecretWord(String[] wordlist, Master master) {
        for(int i=0, match=0; i<10 && match<6; i++) {
            String select = wordlist[new Random().nextInt(wordlist.length)];
            match = master.guess(select);
            List<String> newList = new ArrayList<>();
            for(String word: wordlist) {
                if(check(word, select) == match)
                    newList.add(word);
            }
            wordlist = newList.toArray(new String[newList.size()]);
        }
    }
    
    private int check(String a, String b) {
        int match = 0;
        for(int i=0; i<a.length(); i++) {
            if (a.charAt(i) == b.charAt(i))
                match++;
        }
        return match;
    }
}


/*
	642. Design Search Autocomplete System
	Each TieNode stores a map of (all Strings in this branch) -> (Query Time)

	Notice:
		When building the Trie, Find the next TrieNode and update the node of the NEXT TrieNode !!
*/
class AutocompleteSystem {

    class TrieNode {
        Map<String, Integer> cnt;
        Map<Character, TrieNode> children;
        boolean isWord;
        TrieNode() {
            this.cnt = new HashMap<>();
            this.children = new HashMap<>();
            isWord = false;
        }
    }
    
    String prefix;
    TrieNode root;
    
    public AutocompleteSystem(String[] sentences, int[] times) {
        root = new TrieNode();
        prefix = "";
        for (int i = 0; i < times.length; i++) {
            add(sentences[i], times[i]);
        }
    }
    
    private void add(String s, int n) {
        TrieNode node = root;
        for (char c: s.toCharArray()) {
            TrieNode next = node.children.get(c);
            if (next == null) {
                next = new TrieNode();
                node.children.put(c, next);
            }                
            node = next;
            node.cnt.put(s, node.cnt.getOrDefault(s, 0) + n);                        
        }
        node.isWord = true;
    }
    
    public List<String> input(char c) {
        if (c == '#') {
            add(prefix, 1);
            prefix = "";
            return new ArrayList<String>();
        }
        prefix = prefix + c;
        TrieNode node = root;
        PriorityQueue<Map.Entry<String, Integer>> pq = new PriorityQueue<>((a, b) -> (a.getValue() == b.getValue() ? a.getKey().compareTo(b.getKey()) : b.getValue() - a.getValue()));
        List<String> res = new ArrayList<String>();
        for (char ch: prefix.toCharArray()) {
            if (node.children.get(ch) == null) return new ArrayList<String>();
            node = node.children.get(ch);
        }
        pq.addAll(node.cnt.entrySet());
        for (int i = 0; i < 3 && !pq.isEmpty(); i++) {
            res.add(pq.poll().getKey());
        }
        return res;
    }
}

/*
	1011. Capacity To Ship Packages Within D Days
	Input: weights = [1,2,3,4,5,6,7,8,9,10], D = 5
	Output: 15
	Explanation: 
	A ship capacity of 15 is the minimum to ship all the packages in 5 days like this:
	1st day: 1, 2, 3, 4, 5
	2nd day: 6, 7
	3rd day: 8
	4th day: 9
	5th day: 10

	Notice:
		The LEAST amount! if (check) then r = mid !
*/		
class Solution {
    public int shipWithinDays(int[] weights, int D) {
        int r = 0, l = 0;
        for (int w: weights) {
            l = Math.max(l, w);
            r += w;
        }        
        while (l < r) {
            int mid = (l + r) / 2;
            int d = helper(weights, mid);
            if (d <= D) r = mid;
            else l = mid + 1;
        }
        return l;
    }
    
    private int helper(int[] weights, int n) {        
        int cnt = 0, cur = 0;
        for (int w: weights) {
            if (cur < w) {
                cur = n;
                cnt++;
            }
            cur -= w;
        }
        return cnt;
    }
}	

/*
	315. Count of Smaller Numbers After Self
	Input: [5,2,6,1]
	Output: [2,1,1,0] 
	Explanation:
	To the right of 5 there are 2 smaller elements (2 and 1).
	To the right of 2 there is only 1 smaller element (1).
	To the right of 6 there is 1 smaller element (1).
	To the right of 1 there is 0 smaller element.

	Notice:
		in the srot left and right condition
		smaller[] should add how many right element is in front of this left
*/
class Solution {
    class Pair {
        int val;
        int index;
        Pair (int v, int i) {
            this.val = v;
            this.index = i;
        }
    }
    public List<Integer> countSmaller(int[] nums) {
        Pair[] arr = new Pair[nums.length];
        for (int i = 0; i < nums.length; i++) {
            arr[i] = new Pair(nums[i], i);
        }
        Integer[] smaller = new Integer[nums.length];
        Arrays.fill(smaller, 0);
        List<Integer> res = new ArrayList<>();
        arr = mergeSort(arr, smaller);
        res.addAll(Arrays.asList(smaller));
        return res;
    }
    
    private Pair[] mergeSort(Pair[] arr, Integer[] smaller) {
        if (arr.length <= 1) return arr;
        int mid = arr.length / 2;
        Pair[] left = mergeSort(Arrays.copyOfRange(arr, 0, mid), smaller);
        Pair[] right = mergeSort(Arrays.copyOfRange(arr, mid, arr.length), smaller);
        for (int i = 0, j = 0; i < left.length || j < right.length; ) {
            if (j == right.length || i < left.length && left[i].val <= right[j].val) {
                arr[i+j] = left[i];
                smaller[left[i].index] += j;
                i++;
            } else {
                arr[i+j] = right[j];
                j++;
            }
        }
        return arr;
    }
}


/*
	295. Find Median from Data Stream

	maxQ and minQ, maintain the size of maxQ is equal or one bigger than minQ
	always fistly add into maxQ, if wants the answer to store in the maxQ when size of maxQ and minQ are different
	can choose either maxQ or minQ, just to remember to maintain the size 
*/		
class MedianFinder {

    PriorityQueue<Integer> minQ;
    PriorityQueue<Integer> maxQ;
    
    /** initialize your data structure here. */
    public MedianFinder() {
        minQ = new PriorityQueue<>();
        maxQ = new PriorityQueue<>((a, b) -> (b.compareTo(a)));
    }
    
    public void addNum(int num) {
        maxQ.add(num);
        minQ.add(maxQ.poll());
        if (maxQ.size() < minQ.size())
            maxQ.add(minQ.poll());
    }
    
    public double findMedian() {
        if (minQ.isEmpty() && maxQ.isEmpty()) return 0.0;
        if (minQ.size() != maxQ.size())
            return (double) maxQ.peek();
        else 
            return ( minQ.peek() + maxQ.peek() ) / 2.0;
    }
}	


/*
	第二轮是个韩国小哥，英语特别好让我以为是ABK。
	题目是有一个list of numbers，每一个数代表一张牌，如果我们定义一个顺子(straight)是五个连续的数字，问这个list可不可以完全由顺子组成。
	比如[7,1,2,3,4,3,4,5,6,5]可以由[1,2,3,4,5]和[3,4,5,6,7]组成，所以return True。
	Follow up是如果定义顺子是三张及以上连续的牌，问这个list还可以由顺子组成吗？

	最开始肯定先sort，然后从小到大遍历，用一个dictionary保存当前未完成的顺子（就是长度未达到标准的顺子）的信息。
	key是未完成的顺子的当前最后一个数字是多少，value是一个list，list中每个数字代表所有未完成的顺子中以key为最后一个值的长度是多少。
	举个例子，如果sort之后所有的牌是[1,2,3,3,4,4,5,6,6,7]，
	那么当遍历到第二个4的时候，这个dictionary是 {4:[2,4]}，因为当前有两个未完成的顺子1,2,3,4和3,4，它们都是以4结尾，长度分别为4和2。
	当list中有数字变为5的时候，把这个数字pop出去，最后如果dictionary是空的，就代表成功啦。
*/


/*
	328. Odd Even Linked List

	Notice:
		move odd first then even!!!
*/	
class Solution {
    public ListNode oddEvenList(ListNode head) {
        if (head == null) return head;
        ListNode odd = head, even = head.next, evenHead = even;
        while (even != null && even.next != null) {
            odd.next = odd.next.next;
            even.next = even.next.next;
            odd = odd.next;
            even = even.next;            
        }
        odd.next = evenHead;
        return head;
    }
}


/*
	174. Dungeon Game
	Add a dummy column and row and avoid a lot of edge case code
	initialize the edge slots as max values
	DP from destination to start
*/		
public int calculateMinimumHP(int[][] dungeon) {
    int m = dungeon.length, n = dungeon[0].length;
    int[][] dp = new int[m+1][n+1];
    for(int i=0; i<m; i++)
        dp[i][n] = Integer.MAX_VALUE;
    for(int j=0; j<n; j++)
        dp[m][j] = Integer.MAX_VALUE;
    dp[m-1][n] = 1;
    dp[m][n-1] = 1;
    for(int i=m-1; i>=0; i--) {
        for(int j=n-1; j>=0; j--) {
            int health = Math.min(dp[i+1][j], dp[i][j+1]) - dungeon[i][j];
            dp[i][j] = health <= 0 ? 1 : health;
        }
    }
    return dp[0][0];
}	


/*
	841. Keys and Rooms
	Input: [[1],[2],[3],[]]
	Output: true
	Explanation:  
	We start in room 0, and pick up key 1.
	We then go to room 1, and pick up key 2.
	We then go to room 2, and pick up key 3.
	We then go to room 3.  Since we were able to go to every room, we return true.

	Note: pure DFS or BFS
*/
public boolean canVisitAllRooms(List<List<Integer>> rooms) {
    Queue<Integer> keys = new LinkedList<>();
    Set<Integer> visited = new HashSet<>();
    for (int k: rooms.get(0)) {
        keys.add(k);
    }
    visited.add(0);
    while (!keys.isEmpty()) {
        int r = keys.poll();
        if (!visited.contains(r)) {
            visited.add(r);
            for (int k: rooms.get(r)) {
                keys.add(k);
            }
        }
    }
    return visited.size() == rooms.size();
}	


/*
	815. Bus Routes
	Input: 
	routes = [[1, 2, 7], [3, 6, 7]]
	S = 1
	T = 6
	Output: 2
	Explanation: 
	The best strategy is take the first bus to the bus stop 7, then take the second bus to the bus stop 6.

	Notes:
		Most important part is building the graph!
		Map: stop -> list of buses to the stop
*/
public int numBusesToDestination(int[][] routes, int S, int T) {
    if(S==T) return 0;
    Map<Integer, ArrayList<Integer>> graph = new HashMap<>();
    Set<Integer> set = new HashSet<>();
    // build graph
    for(int i=0; i<routes.length; i++) {
        for(int j=0; j<routes[i].length; j++) {
            ArrayList<Integer> temp = graph.getOrDefault(routes[i][j], new ArrayList<Integer>());
            temp.add(i);
            graph.put(routes[i][j], temp);
        }
    }
    // bfs
    Queue<Integer> q = new LinkedList<>();
    q.offer(S);
    int cnt = 0;
    while(!q.isEmpty()) {
        int size = q.size();
        cnt++;
        for(int i=0; i<size; i++) {
            ArrayList<Integer> buses = graph.get(q.poll());
            for(int bus: buses) {
                if(!set.contains(bus)) {
                    set.add(bus);
                    for(int j=0; j<routes[bus].length; j++) {
                        if(routes[bus][j] == T) return cnt;
                        q.offer(routes[bus][j]);
                    }
                }    
            }                
        }
    }
    return -1;
}


/*
	221. Maximal Square
	Given a 2D binary matrix filled with 0's and 1's, find the largest square containing only 1's and return its area.

	do[i][i] store the biggest length of the square that ends at (i, j)
*/
public int maximalSquare(char[][] matrix) {
    int r=matrix.length;
    if(r==0) return 0;
    int c=matrix[0].length;        
    int[][] dp = new int[r+1][c+1];
    int max = 0;
    for (int i=0; i<r; i++) {
        for (int j=0; j<c; j++) {
            if(matrix[i][j]=='1') {
                dp[i+1][j+1] = Math.min(dp[i][j+1], Math.min(dp[i+1][j], dp[i][j])) + 1;                    
            }
            else {
                    dp[i+1][j+1] = 0;
                }
            max = Math.max(max, dp[i+1][j+1]);
        }
    }
    return max*max;
}	

/*
	Maze Generation
	Recursive backtracker[edit]

	Recursive backtracker on a hexagonal grid
	The depth-first search algorithm of maze generation is frequently implemented using backtracking:

	Make the initial cell the current cell and mark it as visited

	While there are unvisited cells:

		If the current cell has any neighbours which have not been visited:

			Choose randomly one of the unvisited neighbours
			Push the current cell to the stack if it has more than one unvisited neighbor
			Remove the wall between the current cell and the chosen cell
			Make the chosen cell the current cell and mark it as visited

		Else if stack is not empty:

			Pop a cell from the stack while the stack is not empty and the popped cell has no unvisited neighbors
			Make it the current cell

https://en.wikipedia.org/wiki/Maze_generation_algorithm			
*/




/*
	981. Time Based Key-Value Store

	Input: inputs = ["TimeMap","set","get","get","set","get","get"], inputs = [[],["foo","bar",1],["foo",1],["foo",3],["foo","bar2",4],["foo",4],["foo",5]]
	Output: [null,null,"bar","bar",null,"bar2","bar2"]
	Explanation:   
	TimeMap kv;   
	kv.set("foo", "bar", 1); // store the key "foo" and value "bar" along with timestamp = 1   
	kv.get("foo", 1);  // output "bar"   
	kv.get("foo", 3); // output "bar" since there is no value corresponding to foo at timestamp 3 and timestamp 2, then the only value is at timestamp 1 ie "bar"   
	kv.set("foo", "bar2", 4);   
	kv.get("foo", 4); // output "bar2"   
	kv.get("foo", 5); //output "bar2"   

	Notes:
		whenever using TreeMap, REMEBER to check when the floorKey or ceilingKey is null!!!
*/
class TimeMap {

    Map<String, TreeMap<Integer, String>> map;
    
    /** Initialize your data structure here. */
    public TimeMap() {
        map = new HashMap<>();
    }
    
    public void set(String key, String value, int timestamp) {
        if (!map.containsKey(key))
            map.put(key, new TreeMap<Integer, String>());
        TreeMap treeMap = map.get(key);
        treeMap.put(timestamp, value);
        map.put(key, treeMap);
    }
    
    public String get(String key, int timestamp) {
        if (!map.containsKey(key) || map.get(key) == null) return "";
        TreeMap treeMap = map.get(key);
        Integer target = (Integer) treeMap.floorKey(timestamp);
        if (target == null) return "";
        return (String) treeMap.get(target);
    }
}


/*
	743. Network Delay Time

	Input: times = [[2,1,1],[2,3,1],[3,4,1]], N = 4, K = 2
	Output: 2

	Notes:
		Dijkstra, priority queue store every node
*/		
public int networkDelayTime(int[][] times, int N, int K) {
    Map<Integer, Map<Integer, Integer>> map = new HashMap<>();
    Set<Integer> visited = new HashSet<>();
    PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> (a[0] - b[0]));
    for (int[] t: times) {
        map.putIfAbsent(t[0], new HashMap<Integer, Integer>());
        map.get(t[0]).put(t[1], t[2]);
    }
    pq.add(new int[]{0, K});   
    int ans = 0;
    while (!pq.isEmpty()) {
        int[] cur = pq.poll();
        if (visited.contains(cur[1])) continue;
        visited.add(cur[1]);
        ans = Math.max(ans, cur[0]);
        if (visited.size() == N) return ans;
        if (!map.containsKey(cur[1])) continue;
        Map<Integer, Integer> temp = map.get(cur[1]);
        for (int next: temp.keySet()) {
            int cost = temp.get(next);
            pq.add(new int[]{cur[0] + cost, next});
        }
    }
    return -1;
}		


/*
	853. Car Fleet
	Input: target = 12, position = [10,8,0,5,3], speed = [2,4,1,1,3]
	Output: 3
	Explanation:
	The cars starting at 10 and 8 become a fleet, meeting each other at 12.
	The car starting at 0 doesn't catch up to any other car, so it is a fleet by itself.
	The cars starting at 5 and 3 become a fleet, meeting each other at 6.
	Note that no other cars meet these fleets before the destination, so the answer is 3.

	Notes:
		sort by the position from the nearest to the destination to the furthest
		loop and check how long will it takes to arrive at destination
*/
public int carFleet(int target, int[] position, int[] speed) {
    TreeMap<Integer, Double> tm = new TreeMap<>();
    for (int i = 0; i < position.length; i++) 
        tm.put(-position[i], (double) (target - position[i]) / (double) speed[i]);
    int cnt = 0;
    double last = 0;
    for (int p: tm.keySet()) {
        double cur = tm.get(p);            
        if (cur > last) {
            cnt++;
            last = cur;
        }
    }
    return cnt;
}

/*
	380. Insert Delete GetRandom O(1)

	Notes:
		swap the removing val with the last element of list,
		and remove the last element from the list, to acheive O(1) removing

	Follow up: what if allowing duplicating numbers
	instead of storing index as the value of the key in the map, store a set
	which contains every index of that element
	all other basic ideas remain the same
*/
class RandomizedSet {
    
    List<Integer> nums;
    Map<Integer, Integer> map;
    Random rdm;
    
    /** Initialize your data structure here. */
    public RandomizedSet() {
        rdm = new Random();
        nums = new ArrayList<>();
        map = new HashMap<>();
    }
    
    /** Inserts a value to the set. Returns true if the set did not already contain the specified element. */
    public boolean insert(int val) {
        if (map.containsKey(val)) return false;
        map.put(val, nums.size());
        nums.add(val);
        return true;
    }
    
    /** Removes a value from the set. Returns true if the set contained the specified element. */
    public boolean remove(int val) {
        if (!map.containsKey(val)) return false;
        int idx = map.get(val);
        if (idx != nums.size() - 1) {
            int lastNum = nums.get(nums.size() - 1);
            nums.set(idx, lastNum);   
            map.put(lastNum, idx);
        }
        nums.remove(nums.size() - 1);
        map.remove(val);
        return true;
    }
    
    /** Get a random element from the set. */
    public int getRandom() {
        return nums.get(rdm.nextInt(nums.size()));
    }
}

/*
	Follow up code of the question above
*/	
public class RandomizedSet {
	    ArrayList<Integer> nums;
	    HashMap<Integer, Set<Integer>> locs;
	    java.util.Random rand = new java.util.Random();
	    /** Initialize your data structure here. */
	    public RandomizedSet() {
	        nums = new ArrayList<Integer>();
	        locs = new HashMap<Integer, Set<Integer>>();
	    }
	    
	    /** Inserts a value to the set. Returns true if the set did not already contain the specified element. */
	    public boolean insert(int val) {
	        boolean contain = locs.containsKey(val);
	        if ( ! contain ) locs.put( val, new HashSet<Integer>() ); 
	        locs.get(val).add(nums.size());        
	        nums.add(val);
	        return ! contain ;
	    }
	    
	    /** Removes a value from the set. Returns true if the set contained the specified element. */
	    public boolean remove(int val) {
	        boolean contain = locs.containsKey(val);
	        if ( ! contain ) return false;
	        int loc = locs.get(val).iterator().next();
                locs.get(val).remove(loc);
	        if (loc < nums.size() - 1 ) {
	            int lastone = nums.get(nums.size() - 1 );
	            nums.set( loc , lastone );
	            locs.get(lastone).remove(nums.size() - 1);
	            locs.get(lastone).add(loc);
	        }
	        nums.remove(nums.size() - 1);
	        if (locs.get(val).isEmpty()) locs.remove(val);
	        return true;
	    }
	    
	    /** Get a random element from the set. */
	    public int getRandom() {
	        return nums.get( rand.nextInt(nums.size()) );
	    }
	}	


/*
	1101. The Earliest Moment When Everyone Become Friends

	Input: logs = [[20190101,0,1],[20190104,3,4],[20190107,2,3],[20190211,1,5],[20190224,2,4],[20190301,0,3],[20190312,1,2],[20190322,4,5]], N = 6
	Output: 20190301

	Notes:
		Sort the array and union find

	Notice:
		Inside union find's find, parentA = find(parentB) !!		
*/
class Solution {
    
    public class UnionFind {
        int[] parent;
        int size;
        public UnionFind(int n) {
            this.size = n;
            this.parent = new int[n];
            for (int i = 0; i < parent.length; i++)
                parent[i] = i;
        }
        
        public void union(int a, int b) {
            int parentA = find(a);
            int parentB = find(b);
            if (parentA != parentB) {
                size--;
                parent[parentB] = parentA;
            }
        }
        
        public int find(int a) {
            if (parent[a] != a) {
                parent[a] = find(parent[a]);                 
            }
            return parent[a];
        }
    }
    
    public int earliestAcq(int[][] logs, int N) {
        UnionFind unionFind = new UnionFind(N);
        Arrays.sort(logs, (a, b) -> (a[0] - b[0]));
        for (int[] log: logs) {
            unionFind.union(log[1], log[2]);            
            if (unionFind.size == 1)
                return log[0];
        }
        return -1;
    }
}


/*
	410. Split Array Largest Sum
	find a mid, if mid is valid(can split into smaller pieces, return true)
*/
class Solution {
    public int splitArray(int[] nums, int m) {
        long l = 0, r = 0;
        for (int num: nums) {
            l = Math.max(num, l);
            r += num;
        }
        while (l < r) {
            long mid = (l + r) >> 1;
            if (check(mid, m, nums)) 
                r = mid;
            else l = mid + 1;
        }
        return (int)l;
    }
    
    private boolean check(long target, int m, int[] nums) {
        int cur = 0, cnt = 1;
        for (int num: nums) {
            cur += num;
            if (cur > target) {
                cur = num;
                cnt++;
                if (cnt > m) return false;
            }
        }
        return true;
    }
}


/*
	1153. String Transforms Into Another String

	Scan s1 and s2 at the same time,
	record the transform mapping into a map/array.
	The same char should transform to the same char.
	Otherwise we can directly return false.

	To realise the transformation:

	transformation of link link ,like a -> b -> c:
	we do the transformation from end to begin, that is b->c then a->b

	transformation of cycle, like a -> b -> c -> a:
	in this case we need a tmp
	c->tmp, b->c a->b and tmp->a
	Same as the process of swap two variable.

	In both case, there should at least one character that is unused,
	to use it as the tmp for transformation.
	So we need to return if the size of set of unused characters < 26.
*/	
public boolean canConvert(String str1, String str2) {
    if (str1.equals(str2)) return true;
    Map<Character, Character> map = new HashMap<>();
    for (int i = 0; i < str1.length(); i++) {
        char c1 = str1.charAt(i);
        char c2 = str2.charAt(i);
        if (map.containsKey(c1) && map.get(c1) != c2)
            return false;
        map.put(c1, c2);
    }
    return new HashSet<Character>(map.values()).size() < 26;
}	


/*
	658. Find K Closest Elements
	Input: [1,2,3,4,5], k=4, x=-1
	Output: [1,2,3,4]

	find a window of size k, where the left is closer to the target than right, and with the biggest left
*/
public List<Integer> findClosestElements(int[] arr, int k, int x) {
    int l = 0, r = arr.length - k;
    while (l < r) {
        int mid = (l + r) >> 1;
        if (Math.abs(arr[mid] - x) <= Math.abs(arr[mid + k] - x)) r = mid;
        else l = mid + 1;
    }
    List<Integer> ans = new ArrayList<>();
    for (int i = l; i < l + k; i++)
        ans.add(arr[i]);
    return ans;
}	


/*
	403. Frog Jump

	given stones, if a frog jumps k step last time, frog can only jump k, k-1, k+1 for this step
	[0,1,3,5,6,8,12,17]

	There are a total of 8 stones.
	The first stone at the 0th unit, second stone at the 1st unit,
	third stone at the 3rd unit, and so on...
	The last stone at the 17th unit.

	Return true. The frog can jump to the last stone by jumping 
	1 unit to the 2nd stone, then 2 units to the 3rd stone, then 
	2 units to the 4th stone, then 3 units to the 6th stone, 
	4 units to the 7th stone, and 5 units to the 8th stone.

	Notes:
		Map, key is the stone, value is when the frog is on this stone, how many steps can it jump
*/
public boolean canCross(int[] stones) {
    Map<Integer, HashSet<Integer>> map = new HashMap<>();
    for (int i = 0; i < stones.length; i++)
        map.put(stones[i], new HashSet<>());
    map.get(0).add(1);
    for (int s: stones) {
        HashSet<Integer> set = map.get(s);            
        for (int step: set) {
            if (s + step == stones[stones.length-1]) return true;
            if (s + step < stones[stones.length-1]) {
                int next = s + step;
                if (map.containsKey(next)) {
                    HashSet<Integer> nextSet = map.get(next);
                    nextSet.add(step);
                    if (step - 1 > 0) nextSet.add(step - 1);
                    nextSet.add(step + 1);
                }
            }
        }
    }
    return false;
}		


/*
	Skip iterator
*/
class SkipIterator {
	private int index = 0;
	private int[] nums = null;
	private Map<Integer, Integer> map = null;
	public SkipIterator(int[] nums) {
		this.nums = nums;
		map = new HashMap<>();
	}

	/**
	* Returns true if the iteration has more elements.
	*/
	public boolean hasNext() {
		return index < nums.length;
	}

	/**
	* Returns the next element in the iteration.
	*/
	public Integer next() {
		Integer value = nums[index++];
		checkSkipped();
		return value;
	}
	
	private void checkSkipped() {
		while(index < nums.length && map.containsKey(nums[index])) {
			if(map.get(nums[index]) == 1) map.remove(nums[index]);
			else map.put(nums[index], map.get(nums[index])-1);
			++index;
		}
	}

	/**
	* The input parameter is an int, indicating that the next element equals 'num' needs to be skipped.
	* This method can be called multiple times in a row. skip(5), skip(5) means that the next two 5s should be skipped.
	*/ 
	public void skip(int num) {
		map.put(num, map.getOrDefault(num, 0)+1);
		checkSkipped();
	}


}


class NumberCombin {
	int[] memo;
	static public int dfs(int N, int start) {
		if (memo[N] != 0) return memo[N];
		int ret = 0;
		for (int i = start; i <= N; i++) {
			ret += dfs(N - i, i);
		}
		memo[N] = ret;
		return ret;
	}
}


/*
	1055. Shortest Way to Form String

	Greedy match the source to the target
	count the number of time that the source get iterated
*/
public int shortestWay(String source, String target) {
    int res = 0;
    for (int i = 0; i < target.length(); i++) {
        int ancor = i;
        for (int j = 0; j < source.length(); j++) {
            if (i < target.length() && source.charAt(j) == target.charAt(i))
                i++;
        }
        if (ancor == i) return -1;
        i--;
        res++;
    }
    return res;
}	


/*
	334. Increasing Triplet Subsequence

	n1 is the current smallest number, n2 is the second larget number with the subsequence of length 2

	Notice:
		n1 is the smallest, also means it can be "equal" to the smallest!!
*/
public boolean increasingTriplet(int[] nums) {
    if (nums.length < 3) return false;
    Integer n1 = nums[0], n2 = null;
    for (int i = 1; i < nums.length; i++) {
        if (nums[i] <= n1) n1 = nums[i];
        else {
            if (n2 != null && nums[i] > n2) return true;
            n2 = nums[i];
        }
    }
    return false;
}	


/*
	914. X of a Kind in a Deck of Cards

	Count the number of each card
	get the greatest common divisor of each count

	Notice:
		Greatest Common Divisor 
			gcd(a, b) = gcd(b, a mod b)
			gcd(a, 0) = a
*/
class Solution {
    public boolean hasGroupsSizeX(int[] deck) {
        Map<Integer, Integer> map = new HashMap<>();
        int ans = 0;
        for (int i: deck) map.put(i, map.getOrDefault(i, 0) + 1);
        for (int i: map.values()) ans = gcd(i, ans);
        return ans > 1;
    }
    
    private int gcd(int a, int b) {
        return b > 0 ? gcd(b, a % b) : a;   
    }        
}
















