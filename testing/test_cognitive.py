'''
// Example that breaks nesting:
if (x > 0)
{ // Nesting increases here, not on the 'if' line
    // ...
}
The current implementation will score the if on the wrong nesting level if the { is on the next line.
'''