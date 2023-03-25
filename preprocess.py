#!/usr/bin/env python
# coding: utf-8



class Preprocessor:
    def __init__(self, data):
        self.data = data
    
    def preprocess(self):
        input_keys = list(self.data['results']['root_node']['results'].keys())
        for i in input_keys:
            for j in self.data['results']['root_node']['results'][i]['results']:
                yield {
                    'created_at': j['created_at'],
                    'project_node_input_id': j['project_node_input_id'],
                    'project_node_output_id': j['project_node_output_id'],
                    'workpackage_total_size': j['workpackage_total_size'],
                    'loss': j['loss'],

                    # task_output
                    'answer': j['task_output']['answer'], 
                    'cant_solve': j['task_output']['cant_solve'],
                    'corrupt_data': j['task_output']['corrupt_data'],
                    'duration_ms': j['task_output']['duration_ms'],

                    # user
                    'vendor_id': j['user']['vendor_id'],
                    'vendor_user_id': j['user']['vendor_user_id'],
                    'id': j['user']['id'],
                    'image_url': j['task_input']['image_url'],
                }

