# GPT Fewshot Batcher

A basic GUI program to handle conversion fewshots while optimizing for large amounts of text.

I wrote this program to fill a need I saw in AI text management. It's a simple program that provides useful tools for managing fewshots - specifically, conversion fewshots, where you are converting text from one point to another. (such as perspective changes - first person to second person, as an example.) Additionally, I built an advanced context manager from the ground up to handle everything from setting changes, context activation/deactivation, and even positional changes and editing. Everything is built to respect user choice, and stay out of their way as much as possible.

Among its features are these:

* Easy-to-use AI generation and customization
* Full local model support (soon to come), with a download manager for common models
* On the fly model switching (soon to come)
* 'No Model' mode for testing and simple usage
* Advanced context manager
* Highly customizable fewshot formatting
* Mass activation/deactivation (context addition/removal) of any fewshot
* Permanent activation (though this has limited use given the fundamental context limit)
* Context and feature/mode-sensitive fewshot editing
* Fewshot positional shifting
* Saving/loading fewshots

All in under a thousand lines of code!
