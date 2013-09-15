
Uniform lbp
------------------------------------------------------------------------------


     a bitmask is 'uniform' if the number of transitions <= 2.
     
     we precompute the possible values to a lookup table for 
     all possible lbp combinations of n bits(neighbours). 

     check, if the 1st bit eq 2nd, 2nd eq 3rd, ..., last eq 1st, 
       else add a transition for each bit.

       if there's no transition, it's solid
       1 transition: we've found a solid edge.
       2 transitions: we've found a line.

       since the radius of the lbp operator is quite small, 
       we consider any larger number of transitions as noise, 
       and 'discard' them from our histogram, by assinging all of them 
       to a single noise bin


    neighbours   histbins   uniforms
             4         16         14
             5         32         22
             6         64         32
             7        128         44
             8        256         58
             9        512         74
            10       1024         92
            11       2048        112
            12       4096        134
            13       8192        158
            14      16384        184
            15      32768        212
            16      65536        242
            17     131072        274
            18     262144        308
            19     524288        344
            20    1048576        382

    unfortunately, though the 8-bit, 8x8 patch uniform feature 
    uses only 3776 floats instead of 16384 for the full 256 bin histogram
    the yml differs only by a rate of ~0.4.
    
    
    updates:
     ^ ltp and var_lbp
     ^ lfw_funneled database
     ^ combining many small featuresets is quite powerful (combinedLBP), 
       eg, per patch: 
         * 1 3-bit(8 bins) histogram for the pixel center
         * 1 3-bit(8 bins) (8 indices) lpb histogram index of max value
         * 1 4-bit(16 bins) central symmetric (4 corners)   lpb histogram
         * 1 4-bit(16 bins) diagonal tangent (4 corners)   lpb histogram
         concatening those to a 48 bytes patch vector, times 8*8 patches gives 3072 bytes per image ;)
       
     ^ k fold cross validation, please look at results.txt
     ^ added another ref impl, the minmax (surprise)
     ^ this thing has turned into a comparative test of face recognizers
     ^ added a ref impl that just compares pixls
     ^ added uniform version of lbp
