
package org.tartarus.snowball;

import java.lang.reflect.Method;
import java.io.Reader;
import java.io.Writer;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.OutputStream;
import java.io.FileOutputStream;

public class TestApp {
    private static void usage()
    {
        System.err.println("Usage: TestApp <algorithm> <input file> [-o <output file>]");
    }

    public static void main(String [] args) throws Throwable {
	if (args.length < 2) {
            usage();
            return;
        }

	Class stemClass = Class.forName("org.tartarus.snowball.ext." +
					args[0] + "Stemmer");
        SnowballStemmer stemmer = (SnowballStemmer) stemClass.newInstance();
	
	BufferedReader br = new BufferedReader(new InputStreamReader(
					new FileInputStream(args[1]), "UTF-8"));
	//Reader reader;
	//reader = new InputStreamReader(new FileInputStream(args[1]));
	//reader = new BufferedReader(reader);

	//StringBuffer input = new StringBuffer();

        OutputStream outstream;

	if (args.length > 2) {
            if (args.length >= 4 && args[2].equals("-o")) {
                outstream = new FileOutputStream(args[3]);
            } else {
                usage();
                return;
            }
	} else {
	    outstream = System.out;
	}

	BufferedWriter output = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("stem_de.txt"), "UTF-8"));

	//int repeat = 1;
	if (args.length > 4) {
	    //repeat = Integer.parseInt(args[4]);
	}

	Object [] emptyArgs = new Object[0];
	String input_line;
	while ((input_line = br.readLine()) != null) {
			String[] words = input_line.split(" ");
			for (int k = 0; k < words.length; k++) {
				stemmer.setCurrent(words[k]);
				stemmer.stem();
				output.write(stemmer.getCurrent() + " ");
			}

		 //    if (Character.isWhitespace((char) ch)) {
			// if (input.length() > 0) {
			//     stemmer.setCurrent(input.toString());
			//     for (int i = repeat; i != 0; i--) {
			// 	stemmer.stem();
			//     }
			//     output.write(stemmer.getCurrent());
			    
			//     input.delete(0, input.length());
			// }
		 //    } else {
			// 	input.append(Character.toLowerCase(ch));
		 //    }
		    output.write('\n');
	}
	output.flush();
}
}
