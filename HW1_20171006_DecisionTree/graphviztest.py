# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 13:37:23 2017

@author: admin
"""

import graphviz

dot = graphviz.Digraph(comment='The Round Table')

dot.node('A', 'King Arthur')
dot.node('B', 'Sir Bedevere the Wise')
dot.node('L', 'Sir Lancelot the Brave')
dot.edges(['AB', 'AL'])
dot.edge('B', 'L', constraint='false')

dot.render('FileName', view=True)