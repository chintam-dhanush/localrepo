import java.util.*;

public class cns2 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        String key = sc.nextLine().toLowerCase().replaceAll("j", "i");
        String text = sc.nextLine().toLowerCase().replaceAll("j", "i");

        char km[][] = new char[5][5];
        HashSet<Character> h = new HashSet<>();

        int idx = 0;

        // Fill key matrix from keyword
        for (char c : key.toCharArray()) {
            if (h.add(c)) {
                km[idx / 5][idx % 5] = c;
                idx++;
            }
        }

        // Fill remaining letters
        for (char ch = 'a'; ch <= 'z'; ch++) {
            if (ch == 'j') continue;
            if (h.add(ch)) {
                km[idx / 5][idx % 5] = ch;
                idx++;
            }
        }

        System.out.println("key matrix:");
        System.out.println(Arrays.deepToString(km));

        StringBuilder enc = new StringBuilder();

        // Encryption
        for (int i = 0; i < text.length(); i += 2) {
            char a = text.charAt(i);
            char b = (i + 1 < text.length()) ? text.charAt(i + 1) : 'x';

            if (a == b) {
                b = 'x';
                i--;
            }

            int[] p1 = sear(km, a);
            int[] p2 = sear(km, b);

            if (p1[0] == p2[0]) {
                enc.append(km[p1[0]][(p1[1] + 1) % 5]);
                enc.append(km[p2[0]][(p2[1] + 1) % 5]);
            } else if (p1[1] == p2[1]) {
                enc.append(km[(p1[0] + 1) % 5][p1[1]]);
                enc.append(km[(p2[0] + 1) % 5][p2[1]]);
            } else {
                enc.append(km[p1[0]][p2[1]]);
                enc.append(km[p2[0]][p1[1]]);
            }
        }

        System.out.println("Encrypted text:");
        System.out.println(enc.toString());

        // -------- DECRYPTION --------

        String cipher = enc.toString();
        StringBuilder dec = new StringBuilder();

        for (int i = 0; i < cipher.length(); i += 2) {
            char a = cipher.charAt(i);
            char b = cipher.charAt(i + 1);

            int[] p1 = sear(km, a);
            int[] p2 = sear(km, b);

            if (p1[0] == p2[0]) {
                dec.append(km[p1[0]][(p1[1] + 4) % 5]);
                dec.append(km[p2[0]][(p2[1] + 4) % 5]);
            } else if (p1[1] == p2[1]) {
                dec.append(km[(p1[0] + 4) % 5][p1[1]]);
                dec.append(km[(p2[0] + 4) % 5][p2[1]]);
            } else {
                dec.append(km[p1[0]][p2[1]]);
                dec.append(km[p2[0]][p1[1]]);
            }
        }

        System.out.println("Decrypted text:");
        System.out.println(dec.toString());

        sc.close();
    }

    // Search function (fixed syntax only)
    static int[] sear(char km[][], char c) {
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                if (km[i][j] == c) {
                    return new int[]{i, j};
                }
            }
        }
        return null;
    }
}