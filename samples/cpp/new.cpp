void myMethod () {
    try 
    {
        if (condition1) { // +1
            for (int i = 0; i < 10; i++) { // +2 (nesting=1)
                while (condition2) { … } // +3 (nesting=2)
            }
        }
    } catch (ExcepType1 | ExcepType2 e) { // +1
            if (condition2) { … } // +2 (nesting=1)
    }
} 