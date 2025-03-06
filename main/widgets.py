# myapp/widgets.py
from django import forms
from django.utils.safestring import mark_safe

class InlineTinyMCEWidget(forms.Widget):
    def render(self, name, value, attrs=None, renderer=None):
        if value is None:
            value = ''
        html = f"""
            <!-- Toolbar container for persistent toolbar -->
            <div id="toolbar-container-{name}" style="margin-bottom: 5px;"></div>
            <!-- The editor element -->
            <div id="tinymce-editor-{name}" class="editable" contenteditable="true" 
                 style="border: 1px solid #ccc; padding: 10px;">
                {value}
            </div>
            <input type="hidden" id="id_{name}" name="{name}" value="{value}">
            <script>
                document.addEventListener("DOMContentLoaded", function() {{
                    tinymce.init({{
                        selector: "#tinymce-editor-{name}",
                        inline: true,
                        fixed_toolbar_container: "#toolbar-container-{name}",
                        toolbar: "undo redo | bold italic underline | fontsizeselect",
                        menubar: false,
                        // Remove any pre-set readonly flag with a delay
                        init_instance_callback: function(editor) {{
                            setTimeout(function() {{
                                editor.setMode("design");
                                editor.getBody().setAttribute("contenteditable", "true");
                                editor.getContainer().classList.remove("mce-content-readonly");
                            }}, 100);
                        }},
                        setup: function(editor) {{
                            editor.on("init", function() {{
                                editor.focus();
                            }});
                            editor.on("change keyup", function() {{
                                document.getElementById("id_{name}").value = editor.getContent();
                            }});
                        }}
                    }});
                }});
            </script>
        """
        return mark_safe(html)
