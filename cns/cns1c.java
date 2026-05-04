import java.util.*;

public class cns1c {

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        // Input plaintext
        System.out.print("Enter plain text: ");
        String str = sc.nextLine();

        char[] ch = str.toCharArray();
        int n = ch.length;

        int[] a = new int[n];
        int[][] b = new int[n][n];
        int[] c = new int[n];
        char[] d = new char[n];

        // Convert plaintext to numbers
        System.out.println("Plain text matrix is:");
        for (int i = 0; i < n; i++) {
            a[i] = ch[i] - 'a';
            System.out.println(a[i]);
        }

        System.out.println("Enter key:");
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++){
                b[i][j] = sc.nextInt();
                c[i]+=b[i][j]*a[j];
        }
       
        System.out.println("Matrix is:");
        for (int val : c) {
            System.out.println(val);
        }

        // Mod 26
        System.out.println("Matrix after mod is:");
        for (int i = 0; i < n; i++) {
            c[i] = c[i] % 26;
            System.out.println(c[i]);
        }

        // Convert to ciphertext
        System.out.print("Cipher text is: ");
        for (int i = 0; i < n; i++) {
            d[i] = (char) ('a' + c[i]);
            System.out.print(d[i]);
        }
        
        //----------------------------------------------
        //dec
        
        

        int det = b[0][0]*b[1][1] - b[0][1]*b[1][0];
        det = (det % 26 + 26) % 26;
            
        // find modular inverse of determinant
        int invDet = 0;
        for (int i = 1; i < 26; i++)
            if ((det * i) % 26 == 1) invDet = i;
        
        Arrays.fill(c, 0);
        
        int[][] inv = new int[2][2];
        inv[0][0] = b[1][1];
        inv[0][1] = -b[0][1];
        inv[1][0] = -b[1][0];
        inv[1][1] = b[0][0];
        
        // apply mod and multiply with invDet
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++) {
                inv[i][j] = (inv[i][j] * invDet) % 26;
                if (inv[i][j] < 0) inv[i][j] += 26;
            }
        
        // convert d → numeric and multiply
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                c[i] += inv[i][j] * (d[j] - 'a');  // 🔥 key fix
        
        System.out.println("Matrix after mod is:");
        for (int i = 0; i < 2; i++) {
            c[i] = (c[i] % 26 + 26) % 26;
            System.out.println(c[i]);
        }
        
        System.out.print("Decrypted text is: ");
        for (int i = 0; i < 2; i++) {
            d[i] = (char) ('a' + c[i]);
            System.out.print(d[i]);
        }
                
                
        
        sc.close();
    
    }
}